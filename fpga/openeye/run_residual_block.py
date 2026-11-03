#!/usr/bin/env python3
"""Hybrid MAC20 layer1.0 pilot: two OpenEye convolutions, host BN/add/ReLU.

This is not a full-network runtime or an ImageNet accuracy experiment.
Private tensors stay outside the uploaded evidence directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fpga.openeye.prepare_packed_layer_fixture import (
    DEFAULT_PLAN, conv_iohw, load_json, packed_layer, prepare,
    read_float32_le, sha256, verify_source_file,
)
from fpga.openeye.runtime_activations import (
    activation_record, affine_fixed, convolution, residual_add,
)
from fpga.openeye.verify_fpga_bundle import verify_bundle

EXPECTED_OPENEYE_COMMIT = '93f0e82bd6aa0cddee73c9db115884e8ebdd7329'
SHIFT = 24


def digest(array):
    return hashlib.sha256(np.asarray(array, dtype='<i4').tobytes()).hexdigest()


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)


def batchnorm(root, manifest, prefix):
    arrays, identities = {}, {}
    for suffix in ('weight', 'bias', 'running_mean', 'running_var'):
        key = prefix + '.' + suffix
        matches = [r for r in manifest['auxiliary_tensors'] if r['state_key'] == key]
        if len(matches) != 1:
            raise ValueError('BatchNorm auxiliary identity missing/duplicated: ' + key)
        record = matches[0]
        path = verify_source_file(root, record, 'float32')
        arrays[suffix] = np.asarray(read_float32_le(path), dtype=np.float64)
        if list(arrays[suffix].shape) != record['shape']:
            raise ValueError('BatchNorm shape mismatch: ' + key)
        identities[key] = record['sha256']
    if np.any(arrays['running_var'] < 0):
        raise ValueError('Negative BatchNorm running variance')
    alpha = arrays['weight'] / np.sqrt(arrays['running_var'] + 1e-5)
    beta = arrays['bias'] - arrays['running_mean'] * alpha
    return alpha, beta, identities


def positive_scale(values):
    a = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(a)):
        raise ValueError('Nonfinite calibration/reference output')
    return max(float(np.abs(a).max()) / 127, 1e-12)


def bn_coefficients(accumulator, input_scale, weight_scale, alpha, beta, relu):
    real = accumulator * (input_scale * weight_scale) * alpha[:, None, None] + beta[:, None, None]
    scale = positive_scale(np.maximum(real, 0) if relu else real)
    multiplier = np.rint(alpha * input_scale * weight_scale / scale * 2**SHIFT)
    bias = np.rint(beta / scale * 2**SHIFT)
    if not np.all(np.isfinite(multiplier)) or not np.all(np.isfinite(bias)) or max(np.abs(multiplier).max(), np.abs(bias).max()) >= 2**62:
        raise ValueError('BatchNorm coefficients exceed safe fixed-point range')
    return multiplier.astype(np.int64), bias.astype(np.int64), scale


def source_activation(bundle, manifest, size, sample):
    """Verified FP32 vector -> source stem on host -> block input CXY INT8.

    Source prefix is deliberately host FP64, not claimed to be OpenEye RTL.
    Smaller sizes are isolated crop probes with their own zero borders.
    """
    packed = bundle / 'mac20/packed-v1'
    vector_root = bundle / 'mac20/test-vectors-100'
    vectors = load_json(vector_root / 'manifest.json')
    shape = vectors['input_shape']
    if vectors['dtype'] != 'float32' or shape != [100, 3, 224, 224] or not 0 <= sample < shape[0]:
        raise ValueError('Unsupported verified input vector identity')
    vector_path = vector_root / vectors['input']['file']
    if vector_path.parent != vector_root or sha256(vector_path) != vectors['input']['sha256'] or vector_path.stat().st_size != int(np.prod(shape)) * 4:
        raise ValueError('Source input vector checksum/size mismatch')
    image = np.memmap(vector_path, dtype='<f4', mode='r', shape=tuple(shape))[sample].copy()
    if not np.all(np.isfinite(image)):
        raise ValueError('Input vector is nonfinite')
    row = packed_layer(manifest, 'conv1')
    if (row['n'], row['m']) != (4, 4) or row['metadata'] is not None:
        raise ValueError('Host stem reconstruction requires the unchanged dense MAC20 conv1')
    values = read_float32_le(verify_source_file(packed, row['values'], 'float32'))
    weights = np.asarray(conv_iohw(values, row['source_weight_shape'], row['grouping']['grouped_order']))
    alpha, beta, bn_identity = batchnorm(packed, manifest, 'bn1')
    # Source image is CHW; OpenEye/pilot tensors use CXY throughout.
    tensor = np.pad(image.astype(np.float64).transpose(0, 2, 1),
                    ((0, 0), (3, 3), (3, 3)))
    windows = np.lib.stride_tricks.sliding_window_view(tensor, (7, 7), axis=(1, 2))[:, ::2, ::2]
    stem = np.einsum('cxyij,cfji->fxy', windows, weights, optimize=True)
    stem = np.maximum(stem * alpha[:, None, None] + beta[:, None, None], 0)
    padded = np.pad(stem, ((0, 0), (1, 1), (1, 1)), constant_values=-np.inf)
    stem = np.lib.stride_tricks.sliding_window_view(padded, (3, 3), axis=(1, 2))[:, ::2, ::2].max(axis=(-2, -1))
    start = (56 - size) // 2
    crop = stem[:, start:start + size, start:start + size]
    scale = positive_scale(crop)
    q = np.clip(np.rint(crop / scale), 0, 127).astype(np.int64)
    identity = {'vector_manifest_sha256': sha256(vector_root / 'manifest.json'),
                'vector_file_sha256': vectors['input']['sha256'], 'sample_index': sample,
                'vector_seed': vectors.get('seed'), 'dataset': 'packed verification vector; not ImageNet',
                'prefix_execution': 'host FP64 conv1/BN/ReLU/maxpool',
                'stem_weight_sha256': row['values']['sha256'], 'bn_sha256': bn_identity,
                'crop_origin_xy': [start, start], 'spatial_size': size,
                'forwarded_int8_sha256': hashlib.sha256(q.astype('i1').tobytes()).hexdigest()}
    return q, scale, identity


def execute(args):
    bundle = args.bundle_root.resolve()
    private, evidence = args.private_root.resolve(), args.evidence.resolve()
    repo = Path(__file__).resolve().parents[2]
    if private.exists() or evidence.exists() or private == evidence or evidence in private.parents or private in evidence.parents:
        raise ValueError('Private/evidence roots must be new, disjoint directories')
    if repo == private or repo in private.parents or repo == evidence or repo in evidence.parents:
        raise ValueError('Generated tensors/evidence must stay outside the repository')
    ingress = verify_bundle(bundle)
    if not args.reference_only:
        pin = subprocess.check_output(['git', '-C', str(args.openeye_root), 'rev-parse', 'HEAD'], text=True).strip()
        if pin != EXPECTED_OPENEYE_COMMIT or subprocess.check_output(['git', '-C', str(args.openeye_root), 'status', '--short'], text=True).strip():
            raise ValueError('A clean pinned OpenEye checkout is required')
    private.mkdir(parents=True)
    evidence.mkdir(parents=True)
    write_json(evidence / 'ingress-verification.json', ingress)
    packed = bundle / 'mac20/packed-v1'
    manifest = load_json(packed / 'manifest.json')
    q_input, input_scale, input_identity = source_activation(bundle, manifest, args.spatial_size, args.sample)
    original_input = q_input.copy()
    original_scale = input_scale
    plan = load_json(DEFAULT_PLAN)
    for row in plan['layers']:
        if row['layer'] in ('layer1.0.conv1', 'layer1.0.conv2'):
            row['input_width'] = row['input_height'] = args.spatial_size
    plan_path = private / 'pilot-execution-plan.json'
    write_json(plan_path, plan)
    stages = []
    ref_input = q_input.copy()
    for index, layer in enumerate(('layer1.0.conv1', 'layer1.0.conv2')):
        fixture_path = private / f'fixture-{index}'
        fixture_manifest = prepare(packed, layer, fixture_path, execution_plan=plan_path, architecture='full-layer-r3')
        fixture = load_json(fixture_path / 'packed-layer-wrapper.json')
        write_json(evidence / f'fixture-{index}-manifest.json', fixture_manifest)
        activations = activation_record(q_input, input_scale)
        activation_path = private / f'activation-{index}.json'
        write_json(activation_path, activations)
        reference = convolution(ref_input, fixture['weights'])
        alpha, beta, bn_identity = batchnorm(packed, manifest, f'layer1.0.bn{index + 1}')
        m, b, next_scale = bn_coefficients(reference, input_scale,
            fixture_manifest['quantization']['scale'], alpha, beta, relu=index == 0)
        # Freeze from reference calibration BEFORE running the DUT. Learned
        # affine coefficients remain private; evidence contains hashes only.
        coefficients_path = private / f'affine-{index}.json'
        write_json(coefficients_path, {'multiplier': m.tolist(), 'bias': b.tolist(),
                                      'shift': SHIFT, 'output_scale': next_scale})
        actual_path = private / f'accumulator-{index}.bin'
        if args.reference_only:
            actual = reference.copy()
        else:
            env = {**os.environ, 'OPENEYE_ARCHITECTURE': 'full-layer-r3',
                   'OPENEYE_RUNTIME_ACTIVATIONS': str(activation_path),
                   'OPENEYE_RUNTIME_OUTPUT': str(actual_path),
                   'OPENEYE_SIMULATOR': os.environ.get('OPENEYE_SIMULATOR', 'verilator')}
            subprocess.run([str(repo / 'fpga/openeye/run_wrapper_conv_regression.sh'),
                            str(args.openeye_root), str(evidence / f'conv-{index}'),
                            'packed-layer', str(fixture_path)], env=env, check=True)
            expected_bytes = reference.size * 4
            if actual_path.stat().st_size != expected_bytes:
                raise AssertionError('Runtime accumulator output size mismatch')
            actual = np.fromfile(actual_path, dtype='<i4').reshape(reference.shape).astype(np.int64)
        if not np.array_equal(actual, reference):
            raise AssertionError('Residual pilot convolution mismatch: ' + layer)
        q_actual = affine_fixed(actual, m, b, SHIFT, relu=index == 0)
        q_reference = affine_fixed(reference, m, b, SHIFT, relu=index == 0)
        if not np.array_equal(q_actual, q_reference):
            raise AssertionError('Residual pilot BatchNorm/requantization mismatch')
        stages.append({'layer': layer, 'scheme': fixture_manifest['scheme'],
                       'activation_sha256': activations['sha256'], 'input_scale': input_scale,
                       'accumulator_sha256': digest(actual), 'reference_accumulator_sha256': digest(reference),
                       'checked_values': int(actual.size), 'host_bn_sha256': bn_identity,
                       'host_affine_coefficients_sha256': sha256(coefficients_path),
                       'host_affine_shift': SHIFT, 'output_scale': next_scale,
                       'requantized_sha256': digest(q_actual),
                       'forwarded_int8_sha256': hashlib.sha256(q_actual.astype('i1').tobytes()).hexdigest()})
        # Forward DUT-derived activation, not the independently calculated reference.
        q_input, ref_input, input_scale = q_actual, q_reference, next_scale
    if stages[1]['activation_sha256'] != stages[0]['forwarded_int8_sha256']:
        raise AssertionError('Second convolution did not consume first convolution output')
    output_scale = positive_scale(np.maximum(ref_input * input_scale + original_input * original_scale, 0))
    output = residual_add(q_input, input_scale, original_input, original_scale, output_scale)
    reference_output = residual_add(ref_input, input_scale, original_input, original_scale, output_scale)
    if not np.array_equal(output, reference_output):
        raise AssertionError('Residual output mismatch')
    report = {'schema_version': 1, 'result': 'HOST_REFERENCE_ONLY' if args.reference_only else 'PASS',
              'claim_level': 'hybrid-single-residual-block-correctness-not-full-network-not-accuracy',
              'openeye_commit': EXPECTED_OPENEYE_COMMIT, 'domino_commit': subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
              'domino_dirty': bool(subprocess.check_output(['git','-C',str(repo),'status','--short'],text=True).strip()),
              'architecture_id': 'openeye-fpga-full-layer-r3-staging-depths', 'block': 'layer1.0',
              'spatial_size': args.spatial_size, 'full_block_geometry': args.spatial_size == 56,
              'input_identity': input_identity, 'checkpoint_sha256': manifest['checkpoint']['sha256'],
              'scheme_sha256': manifest['scheme']['sha256'],
              'activation_policy': 'single-vector reference-calibrated symmetric INT8; zero point 0',
              'calibration_scope': 'debug correctness only; scales frozen before DUT output',
              'host_ops': ['FP64 source prefix', 'fixed-point BN', 'requantization', 'residual add', 'ReLU'],
              'rtl_ops': [] if args.reference_only else ['layer1.0.conv1', 'layer1.0.conv2'],
              'stages': stages, 'output_scale': output_scale, 'output_sha256': digest(output),
              'reference_output_sha256': digest(reference_output), 'checked_values': int(output.size),
              'eligible_for_network_cost': False, 'accuracy_evaluated': False, 'board_ready': False}
    write_json(evidence / 'residual-block-verification.json', report)
    records = [f for f in sorted(evidence.rglob('*')) if f.is_file() and f.name != 'SHA256SUMS']
    with (evidence / 'SHA256SUMS').open('x') as stream:
        for path in records:
            stream.write(sha256(path) + '  ' + str(path.relative_to(evidence)) + '\n')
    print(json.dumps({k:report[k] for k in ('result','block','spatial_size','checked_values','claim_level')}))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle-root', type=Path, required=True)
    p.add_argument('--openeye-root', type=Path, required=True)
    p.add_argument('--private-root', type=Path, required=True)
    p.add_argument('--evidence', type=Path, required=True)
    p.add_argument('--spatial-size', type=int, choices=(8, 56), default=8)
    p.add_argument('--sample', type=int, default=0)
    p.add_argument('--reference-only', action='store_true')
    execute(p.parse_args())


if __name__ == '__main__':
    main()
