#!/usr/bin/env python3
"""Hybrid ResNet-18 correctness runtime, separate from FP32/INT8 accuracy.

Host source stem; RTL residual convolutions/FC; host BN, partial accumulation,
requantization, residual/ReLU/GAP and FC bias. No full-RTL or board claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fpga.openeye.prepare_packed_layer_fixture import DEFAULT_PLAN, load_json, prepare, read_float32_le, sha256, verify_source_file
from fpga.openeye.run_residual_block import batchnorm, bn_coefficients, digest, positive_scale, source_activation, write_json
from fpga.openeye.runtime_activations import activation_record, affine_fixed, convolution, dense, global_average_pool, residual_add
from fpga.openeye.verify_fpga_bundle import verify_bundle
from fpga.openeye.network_checkpoint import FINAL_SEGMENT, STAGE_SEGMENTS

EXPECTED_OPENEYE_COMMIT = '93f0e82bd6aa0cddee73c9db115884e8ebdd7329'
ROOT = Path(__file__).resolve().parents[2]


def validate_resume_ingress(recorded, current):
    """Compare every verified bundle field except the per-job audit timestamp."""
    if (recorded.keys() != current.keys()
            or any(recorded[field] != current[field] for field in current if field != 'verified_utc')):
        raise ValueError('Checkpoint ingress differs from verified immutable bundle')


def resume_identity(manifest, input_identity, chain_id, packed_manifest_sha256):
    if not chain_id or not chain_id.isdecimal():
        raise ValueError('Resume requires original numeric Actions chain run ID')
    if subprocess.check_output(['git', '-C', str(ROOT), 'status', '--short'], text=True).strip():
        raise ValueError('Resumable network requires clean execution checkout')
    names = ('run_network_inference.py', 'network_checkpoint.py', 'network_layer_tiles.py',
             'runtime_activations.py', 'run_residual_block.py', 'prepare_packed_layer_fixture.py',
             'run_wrapper_conv_regression.sh', 'verify_network_evidence.py')
    return dict(domino_commit=subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
                execution_repository='baichuu/DominoSearch-OpenEye-Runtime',
                chain_id=chain_id, openeye_commit=EXPECTED_OPENEYE_COMMIT,
                architecture='full-layer-r3', checkpoint_sha256=manifest['checkpoint']['sha256'],
                checkpoint_schema_version=2, segment_count=FINAL_SEGMENT + 1,
                scheme_sha256=manifest['scheme']['sha256'], packed_manifest_sha256=packed_manifest_sha256,
                storage_remote=os.environ.get('OPENEYE_NETWORK_PRIVATE_REMOTE', 'domino-drive:DominoSearch-runtime-private'),
                storage_policy=os.environ.get('OPENEYE_NETWORK_PRIVATE_POLICY', 'owner-only'),
                input_identity=input_identity,
                sources={name: sha256(ROOT / 'fpga/openeye' / name) for name in names})


class Runtime:
    def __init__(self, packed, manifest, private, evidence, openeye, reference_only):
        self.packed, self.manifest = packed, manifest
        self.private, self.evidence, self.openeye = private, evidence, openeye
        self.reference_only = reference_only
        self.stages, self.blocks = [], []

    def stage(self, layer, a, ref_a, scale, bn=None, relu=False, rtl=True):
        stage_started = time.perf_counter()
        if not np.array_equal(a, ref_a):
            raise AssertionError('Forwarded activation differs before ' + layer)
        identifier = layer.replace('.', '-')
        plan = load_json(DEFAULT_PLAN)
        for row in plan['layers']:
            if row['layer'] == layer and row['type'] == 'conv2d':
                row['input_width'], row['input_height'] = a.shape[1:]
        plan_path = self.private / f'{identifier}-plan.json'
        write_json(plan_path, plan)
        fixture_root = self.private / f'{identifier}-fixture'
        manifest = prepare(self.packed, layer, fixture_root, execution_plan=plan_path, architecture='full-layer-r3')
        fixture = load_json(fixture_root / 'packed-layer-wrapper.json')
        shutil.copyfile(fixture_root / 'manifest.json', self.evidence / f'{identifier}-fixture-manifest.json')
        linear = fixture['layer_type'] == 'linear'
        reference = dense(ref_a[:, 0, 0], fixture['weights']) if linear else convolution(ref_a, fixture['weights'], fixture['execution']['stride'])
        coefficients = None
        if bn is not None:
            alpha, beta, identity = batchnorm(self.packed, self.manifest, bn)
            m, b, next_scale = bn_coefficients(reference, scale, manifest['quantization']['scale'], alpha, beta, relu)
            coefficients = self.private / f'{identifier}-affine.json'
            write_json(coefficients, dict(multiplier=m.tolist(), bias=b.tolist(), shift=24, output_scale=next_scale))
        input_record = activation_record(a, scale)
        input_path = self.private / f'{identifier}-input.json'
        write_json(input_path, input_record)
        run_rtl = rtl and not self.reference_only
        prepared_at = time.perf_counter()
        if run_rtl:
            output_path = self.private / f'{identifier}-accumulator.bin'
            env = {**{k: v for k, v in os.environ.items() if k not in (
                       'OPENEYE_NETWORK_RESUME_KEY_B64', 'DOMINO_RCLONE_CONFIG_B64')},
                   'OPENEYE_ARCHITECTURE': 'full-layer-r3',
                   'OPENEYE_RUNTIME_ACTIVATIONS': str(input_path), 'OPENEYE_RUNTIME_OUTPUT': str(output_path),
                   'OPENEYE_RUNTIME_CHANNEL_TILES': '1',
                   'OPENEYE_SIMULATOR': os.environ.get('OPENEYE_SIMULATOR', 'verilator')}
            subprocess.run([str(ROOT / 'fpga/openeye/run_wrapper_conv_regression.sh'), str(self.openeye),
                            str(self.evidence / identifier), 'packed-layer', str(fixture_root)], env=env, check=True)
            if output_path.stat().st_size != reference.size * 4:
                raise AssertionError('Runtime output byte count mismatch')
            actual = np.fromfile(output_path, dtype='<i4').reshape(reference.shape).astype(np.int64)
        else:
            actual = reference.copy()
        wrapper_finished_at = time.perf_counter()
        if not np.array_equal(actual, reference):
            raise AssertionError('Full-source accumulator mismatch: ' + layer)
        record = dict(layer=layer, scheme=manifest['scheme'], backend='openeye-rtl' if run_rtl else 'host-reference',
                      input_sha256=input_record['sha256'], input_shape=list(a.shape), input_scale=scale,
                      accumulator_sha256=digest(actual), reference_accumulator_sha256=digest(reference),
                      accumulator_shape=list(actual.shape), checked_values=int(actual.size),
                      weight_scale=manifest['quantization']['scale'])
        record.update(input_values=int(a.size), input_zero_values=int(np.count_nonzero(a == 0)))
        if coefficients is not None:
            actual = affine_fixed(actual, m, b, 24, relu=relu)
            reference = affine_fixed(reference, m, b, 24, relu=relu)
            record.update(bn_sha256=identity, affine_sha256=sha256(coefficients), output_scale=next_scale,
                          output_sha256=digest(actual), forwarded_int8_sha256=hashlib.sha256(actual.astype('i1').tobytes()).hexdigest())
        else:
            next_scale = scale * manifest['quantization']['scale']
        record.update(forwarded_values=int(actual.size),
                      forwarded_zero_values=int(np.count_nonzero(actual == 0)),
                      forwarded_at_positive_limit=int(np.count_nonzero(actual == 127)),
                      preparation_reference_wall_seconds=prepared_at-stage_started,
                      wrapper_and_output_read_wall_seconds=wrapper_finished_at-prepared_at,
                      checking_affine_wall_seconds=time.perf_counter()-wrapper_finished_at,
                      timing_scope='host wall-clock including reference/debug; not FPGA latency')
        self.stages.append(record)
        write_json(self.evidence / f'{identifier}-stage-verification.json', record)
        print(json.dumps(dict(event='stage-pass', **record)), flush=True)
        return actual, reference, next_scale

    def block(self, prefix, a, ref_a, scale, downsample=False, rtl=True):
        original, ref_original, original_scale = a, ref_a, scale
        main, ref_main, main_scale = self.stage(prefix + '.conv1', a, ref_a, scale, prefix + '.bn1', relu=True, rtl=rtl)
        main, ref_main, main_scale = self.stage(prefix + '.conv2', main, ref_main, main_scale, prefix + '.bn2', rtl=rtl)
        if downsample:
            skip, ref_skip, skip_scale = self.stage(prefix + '.downsample.0', original, ref_original, original_scale,
                                                   prefix + '.downsample.1', rtl=rtl)
        else:
            skip, ref_skip, skip_scale = original, ref_original, original_scale
        return self.finish_block(prefix, original, main, ref_main, main_scale,
                                 skip, ref_skip, skip_scale, downsample)

    def finish_block(self, prefix, original, main, ref_main, main_scale,
                     skip, ref_skip, skip_scale, downsample):
        """Join actual and oracle branches using the unchanged residual rule."""
        if main.shape != skip.shape:
            raise AssertionError('Residual branch shape mismatch: ' + prefix)
        out_scale = positive_scale(np.maximum(ref_main * main_scale + ref_skip * skip_scale, 0))
        output = residual_add(main, main_scale, skip, skip_scale, out_scale)
        reference = residual_add(ref_main, main_scale, ref_skip, skip_scale, out_scale)
        if not np.array_equal(output, reference):
            raise AssertionError('Residual add mismatch: ' + prefix)
        self.blocks.append(dict(block=prefix, downsample=downsample, input_shape=list(original.shape),
                                input_sha256=hashlib.sha256(original.astype('i1').tobytes()).hexdigest(),
                                output_shape=list(output.shape), skip_shape=list(skip.shape),
                                forwarded_int8_sha256=hashlib.sha256(output.astype('i1').tobytes()).hexdigest(),
                                output_scale=out_scale, output_sha256=digest(output),
                                reference_sha256=digest(reference), checked_values=int(output.size)))
        return output, reference, out_scale


def run_stage_segment(runtime, segment, a, ref_a, scale, pending):
    """One whole checked layer4 stage; preserve original DUT skip privately."""
    prefix, suffix = STAGE_SEGMENTS[segment]
    if suffix == 'conv1':
        if pending is not None:
            raise ValueError('New block cannot inherit an unfinished skip')
        pending = dict(block=prefix, original_actual=a.tolist(),
                       original_reference=ref_a.tolist(), original_scale=scale)
    if not isinstance(pending, dict) or pending['block'] != prefix:
        raise ValueError('Stage requires its authenticated original skip')
    original = np.asarray(pending['original_actual'], dtype=np.int64)
    ref_original = np.asarray(pending['original_reference'], dtype=np.int64)
    original_scale = pending['original_scale']
    if suffix == 'downsample.0':
        skip, ref_skip, skip_scale = runtime.stage(prefix + '.downsample.0',
            original, ref_original, original_scale, prefix + '.downsample.1', rtl=True)
    else:
        a, ref_a, scale = runtime.stage(prefix + '.' + suffix, a, ref_a, scale,
            prefix + ('.bn1' if suffix == 'conv1' else '.bn2'), relu=suffix == 'conv1', rtl=True)
        if suffix == 'conv1' or prefix == 'layer4.0':
            return a, ref_a, scale, pending
        skip, ref_skip, skip_scale = original, ref_original, original_scale
    a, ref_a, scale = runtime.finish_block(prefix, original, a, ref_a, scale,
        skip, ref_skip, skip_scale, downsample=prefix == 'layer4.0')
    return a, ref_a, scale, None


def execute(args):
    private, evidence, bundle = args.private_root.resolve(), args.evidence.resolve(), args.bundle_root.resolve()
    if private.exists() or evidence.exists() or private == evidence or private in evidence.parents or evidence in private.parents:
        raise ValueError('Require new, disjoint private/evidence directories')
    if any(p == ROOT or ROOT in p.parents for p in (private, evidence)):
        raise ValueError('Generated tensors/evidence must stay outside repository')
    if args.mode in ('full-network', 'fc-smoke') and args.spatial_size != 56:
        raise ValueError('Full-network inference requires original 56x56 stem geometry, not a crop')
    ingress = verify_bundle(bundle)
    if not args.reference_only:
        pin = subprocess.check_output(['git','-C',str(args.openeye_root),'rev-parse','HEAD'],text=True).strip()
        dirty = subprocess.check_output(['git','-C',str(args.openeye_root),'status','--short'],text=True).strip()
        if pin != EXPECTED_OPENEYE_COMMIT or dirty:
            raise ValueError('Require clean pinned OpenEye checkout')
    packed = bundle / 'mac20/packed-v1'
    manifest = load_json(packed / 'manifest.json')
    a, scale, input_identity = source_activation(bundle, manifest, args.spatial_size, args.sample)
    ref_a = a.copy()
    segment = getattr(args, 'segment_index', None)
    context_path = getattr(args, 'resume_context_only', None)
    checkpoint = getattr(args, 'resume_checkpoint', None)
    resumed = None
    identity = None
    if segment is not None or context_path is not None:
        if args.mode != 'full-network' or args.reference_only:
            raise ValueError('Segments are actual full-network RTL, never host-only or crop gates')
        identity = resume_identity(manifest, input_identity, args.chain_id, sha256(packed / 'manifest.json'))
        if context_path is not None:
            write_json(context_path, identity)
            return identity
        if not 0 <= segment <= FINAL_SEGMENT or (segment == 0 and checkpoint is not None) or (segment > 0 and checkpoint is None):
            raise ValueError('Require consecutive network segment and predecessor checkpoint')
    elif checkpoint is not None:
        raise ValueError('Resume checkpoint requires segment mode')
    private.mkdir(parents=True, mode=0o700)
    if checkpoint is not None:
        from fpga.openeye.network_checkpoint import unseal, key_from_environment
        resumed = private / 'authenticated-prefix'
        unseal(checkpoint, resumed, identity, key_from_environment())
    if resumed is None:
        evidence.mkdir(parents=True)
    else:
        from fpga.openeye.network_checkpoint import validate_state, validate_prefix_evidence
        state = load_json(resumed / 'state.json')
        a, ref_a, scale = validate_state(state, identity, segment)
        validate_prefix_evidence(resumed / 'evidence', state)
        shutil.copytree(resumed / 'evidence', evidence)
    if resumed is None:
        write_json(evidence / 'ingress-verification.json', ingress)
    else:
        validate_resume_ingress(load_json(evidence / 'ingress-verification.json'), ingress)
    runtime = Runtime(packed, manifest, private, evidence, args.openeye_root, args.reference_only)
    pending = None
    if resumed is not None:
        runtime.stages, runtime.blocks = state['stages'], state['blocks']
        pending = state.get('pending_block')
    logits = None
    for group in range(1, 5):
        for index in range(2):
            current_segment = (group - 1) * 2 + index
            if segment is not None and (segment >= 6 or current_segment != segment):
                continue
            prefix = f'layer{group}.{index}'
            # A downsample pilot's preceding blocks run on host to produce a
            # genuine preceding activation. Never report them as RTL passes.
            rtl = args.mode == 'full-network' or (args.mode == 'downsample-block' and prefix == 'layer2.0')
            a, ref_a, scale = runtime.block(prefix, a, ref_a, scale, downsample=group > 1 and index == 0, rtl=rtl)
            if args.mode == 'downsample-block' and prefix == 'layer2.0':
                break
        if args.mode == 'downsample-block' and group == 2:
            break
    if segment in STAGE_SEGMENTS:
        a, ref_a, scale, pending = run_stage_segment(runtime, segment, a, ref_a, scale, pending)
    if segment is not None and segment < FINAL_SEGMENT:
        # Reuse only a complete checked block/stage and its separate oracle.
        # An unfinished layer4 block also retains its authenticated original skip.
        from fpga.openeye.network_checkpoint import validate_state, validate_prefix_evidence
        state = dict(schema_version=2, completed_segment=segment, identity=identity,
                     actual=a.tolist(), reference=ref_a.tolist(), scale=scale,
                     stages=runtime.stages, blocks=runtime.blocks, pending_block=pending)
        validate_state(state, identity, segment + 1)
        validate_prefix_evidence(evidence, state)
        write_json(private / 'network-state.json', state)
        summary = dict(result='SEGMENT_PASS', completed_segment=segment,
                       domino_commit=identity['domino_commit'], openeye_commit=EXPECTED_OPENEYE_COMMIT,
                       ingress_verified_utc=ingress['verified_utc'],
                       completed_blocks=[b['block'] for b in runtime.blocks],
                       completed_rtl_stages=[s['layer'] for s in runtime.stages],
                       forwarded_int8_sha256=hashlib.sha256(a.astype('i1').tobytes()).hexdigest(),
                       checkpoint_boundary='checked-stage' if pending else 'checked-block',
                       full_network_complete=False, eligible_for_network_cost=False,
                       accuracy_evaluated=False, board_ready=False)
        write_json(evidence / f'network-segment-{segment:03d}-verification.json', summary)
        print(json.dumps(summary), flush=True)
        return summary
    if args.mode != 'downsample-block':
        if a.shape != (512, 7, 7):
            raise AssertionError('Original ResNet-18 requires 512x7x7 before 7x7 GAP')
        gap, ref_gap = global_average_pool(a), global_average_pool(ref_a)
        if not np.array_equal(gap, ref_gap):
            raise AssertionError('GAP mismatch')
        acc, ref_acc, logit_scale = runtime.stage('fc', gap[:, None, None], ref_gap[:, None, None], scale, rtl=True)
        records = [r for r in manifest['auxiliary_tensors'] if r['state_key'] == 'fc.bias']
        if len(records) != 1 or records[0]['shape'] != [1000]:
            raise ValueError('Source FC bias identity missing')
        bias = np.asarray(read_float32_le(verify_source_file(packed, records[0], 'float32')), dtype=np.float64)
        logits, ref_logits = acc * logit_scale + bias, ref_acc * logit_scale + bias
        if not np.array_equal(logits, ref_logits):
            raise AssertionError('Logits mismatch')
        gap_report = dict(input_shape=list(a.shape), output_shape=[512], rounding='nearest ties to even',
                          input_sha256=hashlib.sha256(a.astype('i1').tobytes()).hexdigest(),
                          output_sha256=digest(gap), forwarded_int8_sha256=hashlib.sha256(gap.astype('i1').tobytes()).hexdigest(),
                          fc_bias_sha256=records[0]['sha256'])
        if gap_report['forwarded_int8_sha256'] != runtime.stages[-1]['input_sha256']:
            raise AssertionError('FC did not consume real GAP output')
    report = dict(schema_version=1, mode=args.mode, result='HOST_REFERENCE_ONLY' if args.reference_only else 'PASS',
                  claim_level='hybrid-resnet18-debug-correctness-not-full-rtl-not-accuracy-not-board',
                  openeye_commit=EXPECTED_OPENEYE_COMMIT,
                  ingress_verified_utc=ingress['verified_utc'],
                  domino_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
                  domino_dirty=bool(subprocess.check_output(['git','-C',str(ROOT),'status','--short'],text=True).strip()),
                  checkpoint_sha256=manifest['checkpoint']['sha256'], scheme_sha256=manifest['scheme']['sha256'],
                  input_identity=input_identity, stages=runtime.stages, blocks=runtime.blocks,
                  rtl_layers=[s['layer'] for s in runtime.stages if s['backend']=='openeye-rtl'],
                  host_ops=['source stem FP64', 'channel partial-sum accumulation', 'fixed BN/requantization', 'residual/ReLU', 'GAP', 'FC bias'],
                  calibration_scope='single-vector debug; each stage coefficients frozen before DUT',
                  eligible_for_network_cost=False, accuracy_evaluated=False, board_ready=False)
    if args.mode == 'full-network' and (len(runtime.stages) != 20 or len(runtime.blocks) != 8
            or (not args.reference_only and len(report['rtl_layers']) != 20)):
        raise AssertionError('Incomplete full-network graph/RTL coverage')
    if logits is not None:
        report.update(gap=gap_report, output_shape=[1000], output_sha256=hashlib.sha256(logits.astype('<f8').tobytes()).hexdigest(),
                      reference_output_sha256=hashlib.sha256(ref_logits.astype('<f8').tobytes()).hexdigest(),
                      top5_class_indices=np.argsort(-logits, kind='stable')[:5].tolist(),
                      top5_note='synthetic verification vector predictions; not accuracy')
    else:
        report.update(output_shape=list(a.shape), output_sha256=digest(a), reference_output_sha256=digest(ref_a))
    write_json(evidence / 'network-verification.json', report)
    files = [p for p in sorted(evidence.rglob('*')) if p.is_file() and p.name != 'SHA256SUMS']
    with (evidence / 'SHA256SUMS').open('x') as stream:
        for path in files:
            stream.write(sha256(path) + '  ' + str(path.relative_to(evidence)) + '\n')
    print(json.dumps({k:report[k] for k in ('mode','result','output_shape','rtl_layers')}), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle-root', type=Path, required=True)
    p.add_argument('--openeye-root', type=Path, required=True)
    p.add_argument('--private-root', type=Path, required=True)
    p.add_argument('--evidence', type=Path, required=True)
    p.add_argument('--mode', choices=('downsample-block','fc-smoke','full-network'), default='downsample-block')
    p.add_argument('--spatial-size', type=int, choices=(16,56), default=56)
    p.add_argument('--sample', type=int, default=0)
    p.add_argument('--reference-only', action='store_true')
    p.add_argument('--segment-index', type=int, choices=range(FINAL_SEGMENT + 1))
    p.add_argument('--resume-checkpoint', type=Path)
    p.add_argument('--chain-id', default='')
    p.add_argument('--resume-context-only', type=Path)
    execute(p.parse_args())


if __name__ == '__main__':
    main()
