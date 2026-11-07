#!/usr/bin/env python3
"""Fail-closed hybrid runtime gate: full checksums, provenance and real JUnit.

Does not recompute private tensors or certify accuracy/latency/board readiness.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fpga.openeye.openeye_evidence_bundle import file_record, load_json, parse_key_values, parse_sha256s, safe_relative
from fpga.openeye.network_layer_tiles import channel_filter_plan
from fpga.openeye.run_network_inference import EXPECTED_OPENEYE_COMMIT
from fpga.openeye.prepare_packed_layer_fixture import DEFAULT_PLAN
from fpga.openeye.network_model_source import selected_model, execution_plan, model_ingress, DENSE_PROFILE, DENSE_IDENTITIES
from fpga.openeye.summarize_wrapper_metrics import parse_log, validate_records

SOURCE_ROOT = Path(__file__).resolve().parent
REQUIRED_SOURCES = {
    'activation_packing_patch_sha256': 'patches/openeye-activation-packing-compat.patch',
    'serial_odd_channel_padding_patch_sha256': 'patches/openeye-serial-odd-channel-padding.patch',
    'layer_mapper_shared_state_patch_sha256': 'patches/openeye-layer-mapper-shared-state.patch',
    'network_tile_tool_sha256': 'network_layer_tiles.py',
    'runtime_activation_tool_sha256': 'runtime_activations.py',
    'runtime_activation_patch_sha256': 'patches/openeye-wrapper-runtime-activations.patch',
}


def verify_layer(root, expected_stage=None):
    report = load_json(root / 'spatial_layer_verification.json')
    plan = channel_filter_plan(report['channel_count'], report['filter_count'])
    if report['rtl_correctness'] != 'PASS' or [t['tile'] for t in report['tiles']] != plan:
        raise ValueError('Incomplete channel/filter tile coverage')
    if report['output_sha256'] != report['reference_sha256'] or report['eligible_for_network_cost'] or report.get('board_ready', False):
        raise ValueError('Wrong output reference or runtime cost claim')
    junit_paths = sorted((root / 'spatial-tiles').glob('junit-*.xml'))
    if len(junit_paths) != len(plan):
        raise ValueError('Missing or extra real JUnit results')
    for index, (tile, path) in enumerate(zip(report['tiles'], junit_paths)):
        if path.name != f'junit-{index:04d}.xml' or tile != load_json(root / 'spatial-tiles' / f'tile-{index:04d}.json'):
            raise ValueError('Tile identity or JUnit sequence mismatch')
        cases = list(ET.parse(path).iter('testcase'))
        if len(cases) != 1 or any(c.find(tag) is not None for c in cases for tag in ('failure','error','skipped')):
            raise ValueError('Failed/error/skipped JUnit gate')
        if tile['result'] != 'PASS' or tile['output_sha256'] != tile['reference_sha256']:
            raise ValueError('Failed tile value gate')
    metrics = load_json(root / 'hardware_metrics.json')
    if metrics['validation']['result'] != 'PASS' or metrics['rtl_correctness'] != 'PASS' or metrics['eligible_for_cost_table']:
        raise ValueError('Invalid counters/correctness-only scope')
    if len(metrics['records']) != len(plan):
        raise ValueError('Missing partial execution counters')
    raw_records, raw_counts = parse_log(root / 'wrapper-conv-packed-layer.log')
    if (raw_counts != metrics['host_section_counts']
            or [dict(r, scheme=m['scheme']) for r, m in zip(raw_records, metrics['records'])] != metrics['records']
            or len(raw_records) != len(plan)
            or validate_records(metrics['records'], raw_counts, [r['scheme'] for r in metrics['records']])):
        raise ValueError('Counter evidence does not match accepted-handshake log')
    provenance = parse_key_values(root / 'regression_manifest.txt')
    if provenance['openeye_commit'] != EXPECTED_OPENEYE_COMMIT or provenance['result'] != 'PASS' or provenance['runtime_channel_tiles'] != '1':
        raise ValueError('Wrong pin/backend or failed RTL provenance')
    if provenance['activation_payload'] != 'verified-runtime-int8-cxy':
        raise ValueError('Stale constant activation label')
    if provenance['architecture_id'] != 'openeye-fpga-full-layer-r3-staging-depths' or provenance['cluster_rows'] != '3':
        raise ValueError('Wrong runtime hardware architecture')
    for key in ('r3_weight_send_tail_patch_sha256', 'iact_write_alignment_patch_sha256'):
        if provenance.get(key) != 'not-applied-upstream':
            raise ValueError('Superseded patch must be supplied by pinned upstream: ' + key)
    for key, relative in REQUIRED_SOURCES.items():
        if provenance.get(key) != file_record(SOURCE_ROOT, Path(relative))['sha256']:
            raise ValueError('Wrong maintained helper/patch identity: ' + key)
    projection = report['logical_execution'].get('kernel_width') == 1
    expected_projection_patch = (file_record(SOURCE_ROOT, Path('patches/openeye-runtime-projection-dense-iact.patch'))['sha256']
                                 if projection else 'not-applied')
    if (provenance.get('runtime_projection_iact_patch_sha256') != expected_projection_patch
            or report.get('projection_activation_storage') != ('all positions including zeros' if projection else 'default')):
        raise ValueError('Wrong projection activation storage/patch identity')
    kernel3 = report['logical_execution'].get('kernel_width') == 3
    expected_kernel3_patch = (file_record(SOURCE_ROOT, Path('patches/openeye-runtime-kernel3-retained-iact.patch'))['sha256']
                             if kernel3 else 'not-applied')
    if (provenance.get('runtime_kernel3_iact_patch_sha256') != expected_kernel3_patch
            or report.get('kernel3_activation_storage') != ('all positions including zeros' if kernel3 else 'default')):
        raise ValueError('Wrong kernel3 activation storage/patch identity')
    if expected_stage is not None:
        if (expected_stage['input_sha256'] != report['activation_sha256']
                or expected_stage['accumulator_sha256'] != report['output_sha256']
                or expected_stage['reference_accumulator_sha256'] != report['output_sha256']
                or expected_stage['checked_values'] != report['checked_values']):
            raise ValueError('Stage output/input hash or shape count mismatch')
    return len(plan)


def verify(root, mode, domino_commit):
    if root.is_symlink() or any(p.is_symlink() for p in root.rglob('*')):
        raise ValueError('Symlink in network evidence')
    checksums = parse_sha256s(root / 'SHA256SUMS')
    expected_files = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name != 'SHA256SUMS'}
    actual_files = set()
    for relative, digest in checksums.items():
        path = safe_relative(relative, 'network checksum')
        actual_files.add(str(path))
        if file_record(root, path)['sha256'] != digest:
            raise ValueError('Network artifact checksum mismatch: ' + relative)
    if actual_files != expected_files or len(actual_files) != len(checksums):
        raise ValueError('Uncovered or missing evidence files')
    if any(p.suffix in ('.bin','.tar','.pt','.pth','.npy','.npz') for p in root.rglob('*')):
        raise ValueError('Private tensor/checkpoint payload in evidence')
    report = load_json(root / 'network-verification.json')
    if (report['result'] != 'PASS' or report['mode'] != mode or report['domino_dirty']
            or report['domino_commit'] != domino_commit or report['openeye_commit'] != EXPECTED_OPENEYE_COMMIT):
        raise ValueError('Wrong report result/scope/pin/clean execution identity')
    if report['eligible_for_network_cost'] or report['accuracy_evaluated'] or report['board_ready']:
        raise ValueError('Invalid latency/accuracy/board claim')
    if mode == 'downsample-block' and report['input_identity']['spatial_size'] not in (16, 56):
        raise ValueError('Unsupported downsample input geometry')
    ingress = load_json(root / 'ingress-verification.json')
    profile = selected_model(report.get('model_profile', 'mac20'))
    model_source = model_ingress(ingress, profile)
    if model_source['checkpoint_sha256'] != report['checkpoint_sha256']:
        raise ValueError('Wrong immutable checkpoint identity')
    if profile == DENSE_PROFILE and report['scheme_sha256'] != DENSE_IDENTITIES['scheme_sha256']:
        raise ValueError('Wrong dense source scheme')
    expected_layers = ({'layer2.0.conv1','layer2.0.conv2','layer2.0.downsample.0'} if mode=='downsample-block' else
                       {'fc'} if mode=='fc-smoke' else
                       {r['layer'] for r in execution_plan(profile)['layers'] if r['layer']!='conv1'})
    if len(report['rtl_layers']) != len(expected_layers) or set(report['rtl_layers']) != expected_layers:
        raise ValueError('Missing or unexpected RTL layer coverage')
    stages = {s['layer']: s for s in report['stages']}
    if len(stages) != len(report['stages']):
        raise ValueError('Duplicate network stage')
    expected_order = []
    expected_blocks = [f'layer{g}.{i}' for g in range(1,5) for i in range(2)]
    if mode == 'downsample-block':
        expected_blocks = expected_blocks[:3]
    forwarded = report['input_identity']['forwarded_int8_sha256']
    for prefix, block in zip(expected_blocks, report['blocks']):
        expected_order += [prefix + '.conv1', prefix + '.conv2']
        downsample = prefix in ('layer2.0','layer3.0','layer4.0')
        if downsample:
            expected_order.append(prefix + '.downsample.0')
        first, second = stages[prefix + '.conv1'], stages[prefix + '.conv2']
        if (block['block'] != prefix or block['downsample'] != downsample
                or block['input_sha256'] != forwarded or first['input_sha256'] != forwarded
                or first['forwarded_int8_sha256'] != second['input_sha256']
                or block['output_sha256'] != block['reference_sha256']
                or block['output_shape'] != block['skip_shape']
                or block['input_shape'] != first['input_shape']
                or block['output_shape'] != second['accumulator_shape']):
            raise ValueError('Broken residual/activation chain')
        if downsample and (stages[prefix + '.downsample.0']['input_sha256'] != forwarded
                or stages[prefix + '.downsample.0']['accumulator_shape'] != block['output_shape']):
            raise ValueError('Downsample did not consume original skip input')
        forwarded = block['forwarded_int8_sha256']
    if mode != 'downsample-block':
        expected_order.append('fc')
        if report['gap']['input_sha256'] != forwarded:
            raise ValueError('GAP did not consume final residual output')
    if ([s['layer'] for s in report['stages']] != expected_order
            or len(report['blocks']) != len(expected_blocks)):
        raise ValueError('Incomplete or reordered network graph')
    source_plan = {r['layer']: r for r in execution_plan(profile)['layers']}
    junit = 0
    for layer, stage in stages.items():
        source = load_json(root / (layer.replace('.','-') + '-fixture-manifest.json'))
        if (source['source_layer'] != layer or source['source_checkpoint_sha256'] != report['checkpoint_sha256']
                or source['source_scheme_sha256'] != report['scheme_sha256'] or source['scheme'] != stage['scheme']
                or source['source_packed_manifest_sha256'] != model_source['packed_manifest_sha256']):
            raise ValueError('Wrong source checkpoint/scheme/layer identity')
        expected_backend = 'openeye-rtl' if layer in expected_layers else 'host-reference'
        if stage['backend'] != expected_backend:
            raise ValueError('Wrong stage backend')
        if load_json(root / (layer.replace('.','-') + '-stage-verification.json')) != stage:
            raise ValueError('Stage evidence differs from network report')
        execution = source['execution']
        original = source_plan[layer]
        if (source['source_weight_shape'] != original['weight_shape']
                or stage['scheme'] != original['expected_scheme']):
            raise ValueError('Source weight shape or mixed-N:M scheme altered')
        original = dict(original)
        if layer != 'fc':
            f, c, h, w = original['weight_shape']
            original.update(input_channels=c, output_channels=f, kernel_width=w, kernel_height=h)
        dimensions = ('input_features','output_features') if layer=='fc' else (
            'input_channels','output_channels','kernel_width','kernel_height','stride')
        if any(execution[k] != original[k] for k in dimensions):
            raise ValueError('Source execution dimensions altered')
        expected_input = ([execution['input_features'],1,1] if layer=='fc' else
                          [execution['input_channels'],execution['input_width'],execution['input_height']])
        expected_output = ([execution['output_features']] if layer=='fc' else
                           [execution['output_channels'],(execution['input_width']+execution['stride']-1)//execution['stride'],
                            (execution['input_height']+execution['stride']-1)//execution['stride']])
        if (stage['input_shape'] != expected_input or stage['accumulator_shape'] != expected_output
                or stage['checked_values'] != math.prod(expected_output)):
            raise ValueError('Wrong full-source stage geometry')
        if mode != 'downsample-block' and layer != 'fc' and any(
                execution[k] != original[k] for k in ('input_width','input_height')):
            raise ValueError('Cropped full-network source geometry')
        if mode == 'downsample-block':
            spatial = report['input_identity']['spatial_size']
            expected_spatial = spatial // 2 if layer == 'layer2.0.conv2' else spatial
            if any(execution[k] != expected_spatial for k in ('input_width', 'input_height')):
                raise ValueError('Downsample stage geometry differs from declared input')
    for layer in expected_layers:
        stage = stages[layer]
        if stage['backend'] != 'openeye-rtl':
            raise ValueError('Reference output masquerades as RTL')
        layer_report = load_json(root / layer.replace('.','-') / 'spatial_layer_verification.json')
        source = load_json(root / (layer.replace('.','-') + '-fixture-manifest.json'))
        provenance = parse_key_values(root / layer.replace('.','-') / 'regression_manifest.txt')
        if (layer_report['logical_execution'] != source['execution']
                or layer_report['channel_count'] != stage['input_shape'][0]
                or layer_report['filter_count'] != stage['accumulator_shape'][0]
                or provenance['fixture_manifest_sha256'] != file_record(root, Path(layer.replace('.','-') + '-fixture-manifest.json'))['sha256']):
            raise ValueError('Partial coverage differs from verified source fixture')
        junit += verify_layer(root / layer.replace('.','-'), stage)
    if report['output_sha256'] != report['reference_output_sha256']:
        raise ValueError('Final runtime output mismatch')
    if mode == 'downsample-block' and (report['output_sha256'] != report['blocks'][-1]['output_sha256']
            or report['output_shape'] != report['blocks'][-1]['output_shape']):
        raise ValueError('Pilot output differs from final residual output')
    if mode in ('fc-smoke','full-network'):
        if report['input_identity']['spatial_size'] != 56 or report['output_shape'] != [1000] or len(report['blocks']) != 8:
            raise ValueError('Incomplete/cropped network geometry')
        if report['gap']['forwarded_int8_sha256'] != stages['fc']['input_sha256']:
            raise ValueError('FC input is not real GAP output')
    focused = sorted(root.glob('focused-*'))
    if len(focused) != 6:
        raise ValueError('Missing focused network gates')
    focused_junit = sum(verify_layer(p) for p in focused)
    if focused_junit != 8:
        raise ValueError('Unexpected focused coverage')
    signatures = set()
    for p in focused:
        execution = load_json(p / 'spatial_layer_verification.json')['logical_execution']
        if 'input_features' in execution:
            signature = ('linear', execution['input_features'],execution['output_features'])
        else:
            signature = ('conv2d',execution['input_channels'],execution['output_channels'],
                         execution['input_width'],execution['input_height'],
                         execution['kernel_width'],execution['kernel_height'],execution['stride'])
        signatures.add(signature)
    if signatures != {('linear',128,8),('conv2d',128,8,8,8,3,3,2),
                      ('conv2d',64,8,8,8,1,1,2),('conv2d',64,8,7,7,3,3,1),
                      ('conv2d',64,8,16,16,1,1,2),('conv2d',64,8,16,16,3,3,2)}:
        raise ValueError('Wrong focused stride/projection/border/Dense coverage')
    return dict(result='PASS', mode=mode, file_checksums=len(checksums), rtl_junit=junit,
                focused_junit=focused_junit, domino_commit=domino_commit,
                openeye_commit=EXPECTED_OPENEYE_COMMIT)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence', type=Path, required=True)
    p.add_argument('--mode', choices=('downsample-block','fc-smoke','full-network'), required=True)
    p.add_argument('--domino-commit', required=True)
    args = p.parse_args()
    print(json.dumps(verify(args.evidence, args.mode, args.domino_commit), sort_keys=True))


if __name__ == '__main__':
    main()
