"""Metadata gate tests use fabricated records, never claimed as RTL evidence."""
import hashlib
import json
import math
from pathlib import Path

import pytest
import yaml

from fpga.openeye.network_layer_tiles import channel_filter_plan
from fpga.openeye.prepare_packed_layer_fixture import DEFAULT_PLAN, load_json
from fpga.openeye.verify_network_evidence import EXPECTED_OPENEYE_COMMIT, REQUIRED_SOURCES, SOURCE_ROOT, verify, verify_layer


DIGEST = 'a' * 64
COMMIT = 'b' * 40


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def seal(root):
    files = sorted(p for p in root.rglob('*') if p.is_file() and p.name != 'SHA256SUMS')
    (root / 'SHA256SUMS').write_text(''.join(
        hashlib.sha256(p.read_bytes()).hexdigest() + '  ' + str(p.relative_to(root)) + '\n' for p in files))


def layer_evidence(root, channels, filters, execution=None, count=1, fixture_hash=DIGEST):
    plan = channel_filter_plan(channels, filters)
    tiles = [dict(tile=t, result='PASS', output_sha256=DIGEST, reference_sha256=DIGEST) for t in plan]
    report = dict(channel_count=channels, filter_count=filters, tiles=tiles, rtl_correctness='PASS',
                  output_sha256=DIGEST, reference_sha256=DIGEST, activation_sha256=DIGEST,
                  logical_execution=execution or {}, eligible_for_network_cost=False, checked_values=count)
    projection = (execution or {}).get('kernel_width') == 1
    report['projection_activation_storage'] = 'all positions including zeros' if projection else 'default'
    kernel3 = (execution or {}).get('kernel_width') == 3
    report['kernel3_activation_storage'] = 'all positions including zeros' if kernel3 else 'default'
    write(root / 'spatial_layer_verification.json', report)
    raw = dict(dma_bitwidth=64, total_cycles=10, load_cycles=4, compute_cycles=6,
               input_accepted_words=4, accepted_weight_words=1, accepted_weight_bytes=8,
               output_accepted_words=1, input_stall_cycles=0, output_stall_cycles=0, stall_cycles=0,
               state_cycles={'GET_WGHT': 4, 'WAIT_FOR_RESULTS': 6},
               input_accepted_words_by_state={'GET_WGHT': 1, 'GET_IACT': 3})
    counts = dict(iact=3, wght=1, psum=0, quantize=0)
    write(root / 'hardware_metrics.json', dict(validation=dict(result='PASS'), rtl_correctness='PASS',
          eligible_for_cost_table=False, records=[dict(raw, scheme='4:4') for _ in plan],
          host_section_counts=[counts for _ in plan]))
    (root / 'wrapper-conv-packed-layer.log').write_text(
        ('Host will send: iact 3, wght 1, psum 0, quantize 0 words\nDOMINO_HW_METRICS ' + json.dumps(raw) + '\n') * len(plan))
    (root / 'regression_manifest.txt').write_text(
        f'openeye_commit={EXPECTED_OPENEYE_COMMIT}\nresult=PASS\nruntime_channel_tiles=1\n'
        f'activation_payload=verified-runtime-int8-cxy\nfixture_manifest_sha256={fixture_hash}\n'
        'architecture_id=openeye-fpga-full-layer-r3-staging-depths\ncluster_rows=3\n'
        'r3_weight_send_tail_patch_sha256=not-applied-upstream\n'
        'iact_write_alignment_patch_sha256=not-applied-upstream\n' + ''.join(
            key + '=' + hashlib.sha256((SOURCE_ROOT / relative).read_bytes()).hexdigest() + '\n'
            for key, relative in REQUIRED_SOURCES.items()) +
        'runtime_projection_iact_patch_sha256=' + (hashlib.sha256(
            (SOURCE_ROOT / 'patches/openeye-runtime-projection-dense-iact.patch').read_bytes()).hexdigest()
            if projection else 'not-applied') + '\n' +
        'runtime_kernel3_iact_patch_sha256=' + (hashlib.sha256(
            (SOURCE_ROOT / 'patches/openeye-runtime-kernel3-retained-iact.patch').read_bytes()).hexdigest()
            if kernel3 else 'not-applied') + '\n')
    for index, tile in enumerate(tiles):
        write(root / 'spatial-tiles' / f'tile-{index:04d}.json', tile)
        (root / 'spatial-tiles' / f'junit-{index:04d}.xml').write_text(
            '<testsuites><testsuite><testcase name="fabricated-metadata-unit-test"/></testsuite></testsuites>')


def network_evidence(tmp_path, mode='downsample-block', spatial_size=16):
    root = tmp_path / 'evidence'
    root.mkdir()
    plan = {r['layer']: r for r in load_json(DEFAULT_PLAN)['layers']}
    rtl = (['layer2.0.conv1','layer2.0.conv2','layer2.0.downsample.0'] if mode=='downsample-block' else
           ['fc'] if mode=='fc-smoke' else [layer for layer in plan if layer!='conv1'])
    stages, blocks = [], []
    size, channels = (spatial_size if mode=='downsample-block' else 56), 64
    prefixes = ([f'layer{g}.{i}' for g in range(1,5) for i in range(2)] if mode!='downsample-block' else
                ['layer1.0','layer1.1','layer2.0'])
    for prefix in prefixes:
        downsample = prefix in ('layer2.0','layer3.0','layer4.0')
        layers = [prefix + '.conv1', prefix + '.conv2'] + ([prefix + '.downsample.0'] if downsample else [])
        original_shape = [channels,size,size]
        for layer in layers:
            row = plan[layer]
            f, c, h, w = row['weight_shape']
            in_size = original_shape[1] if not layer.endswith('conv2') else size
            execution = dict(input_channels=c,output_channels=f,kernel_width=w,kernel_height=h,
                             input_width=in_size,input_height=in_size,stride=row['stride'])
            out_size = (in_size + row['stride'] - 1) // row['stride']
            shape = [f,out_size,out_size]
            stage = dict(layer=layer, scheme=row['expected_scheme'], backend='openeye-rtl' if layer in rtl else 'host-reference',
                         input_sha256=DIGEST,input_shape=[c,in_size,in_size], accumulator_shape=shape,
                         accumulator_sha256=DIGEST,reference_accumulator_sha256=DIGEST,
                         checked_values=math.prod(shape),forwarded_int8_sha256=DIGEST)
            stages.append(stage)
            slug = layer.replace('.','-')
            manifest_path = root / (slug + '-fixture-manifest.json')
            write(manifest_path, dict(source_layer=layer,source_checkpoint_sha256=DIGEST,
                  source_packed_manifest_sha256=DIGEST,
                  source_scheme_sha256=DIGEST,scheme=row['expected_scheme'],execution=execution,
                  source_weight_shape=row['weight_shape']))
            write(root / (slug + '-stage-verification.json'), stage)
            if layer in rtl:
                layer_evidence(root / slug,c,f,execution,math.prod(shape),hashlib.sha256(manifest_path.read_bytes()).hexdigest())
            if layer.endswith('conv1'):
                size, channels = out_size, f
        blocks.append(dict(block=prefix,downsample=downsample,input_sha256=DIGEST,input_shape=original_shape,
                           output_shape=[channels,size,size],skip_shape=[channels,size,size],
                           output_sha256=DIGEST,reference_sha256=DIGEST,forwarded_int8_sha256=DIGEST))
    if mode != 'downsample-block':
        execution = dict(input_features=512,output_features=1000,dataflow='row_stationary')
        stage = dict(layer='fc',scheme='4:4',backend='openeye-rtl',input_sha256=DIGEST,input_shape=[512,1,1],
                     accumulator_shape=[1000],accumulator_sha256=DIGEST,reference_accumulator_sha256=DIGEST,
                     checked_values=1000)
        stages.append(stage)
        manifest_path = root / 'fc-fixture-manifest.json'
        write(manifest_path,dict(source_layer='fc',source_checkpoint_sha256=DIGEST,source_scheme_sha256=DIGEST,
              source_packed_manifest_sha256=DIGEST,
              scheme='4:4',execution=execution,source_weight_shape=[1000,512]))
        write(root / 'fc-stage-verification.json',stage)
        layer_evidence(root / 'fc',512,1000,execution,1000,hashlib.sha256(manifest_path.read_bytes()).hexdigest())
    report = dict(result='PASS',mode=mode,domino_dirty=False,domino_commit=COMMIT,
                  openeye_commit=EXPECTED_OPENEYE_COMMIT,eligible_for_network_cost=False,accuracy_evaluated=False,
                  board_ready=False,checkpoint_sha256=DIGEST,scheme_sha256=DIGEST,
                  input_identity=dict(spatial_size=spatial_size if mode=='downsample-block' else 56,forwarded_int8_sha256=DIGEST),
                  rtl_layers=rtl, stages=stages,blocks=blocks,output_shape=[128,spatial_size//2,spatial_size//2] if mode=='downsample-block' else [1000],
                  output_sha256=DIGEST,reference_output_sha256=DIGEST)
    if mode != 'downsample-block':
        report['gap'] = dict(input_sha256=DIGEST,forwarded_int8_sha256=DIGEST)
    write(root / 'network-verification.json',report)
    write(root / 'ingress-verification.json',dict(method='packed-v1-fpga-ingress-integrity',source_commit='3060cf7',
          candidates=dict(mac20=dict(checkpoint_sha256=DIGEST,packed_manifest_sha256=DIGEST))))
    for index, c in enumerate((128,64,64,128,64,64)):
        execution = (dict(input_features=128,output_features=8) if index==3 else
                     dict(input_channels=c,output_channels=8,input_width=7 if index==2 else 16 if index in (4,5) else 8,
                          input_height=7 if index==2 else 16 if index in (4,5) else 8,kernel_width=1 if index in (1,4) else 3,
                          kernel_height=1 if index in (1,4) else 3,stride=1 if index==2 else 2))
        layer_evidence(root / f'focused-{index}',c,8,execution)
    seal(root)
    return root


@pytest.fixture
def downsample_evidence(tmp_path):
    return network_evidence(tmp_path)


def test_report_exports_measured_counters_without_focused_or_host_counts(downsample_evidence, tmp_path):
    from fpga.openeye.export_network_measurements import collect, write_report
    result = collect(downsample_evidence, COMMIT)
    assert result['verified_rtl_layers'] == 3 and result['unique_rtl_junit'] == 64
    assert result['expected_rtl_layers'] == 20 and not result['full_network_verified']
    assert result['totals']['total_cycles'] == 640
    assert result['totals']['accepted_weight_bytes'] == 512
    assert result['totals']['input_accepted_bytes'] == 2048
    assert result['totals']['output_accepted_bytes'] == 512
    assert len([r for r in result['layers'] if r['status']=='NO_COMPLETE_RTL_EVIDENCE']) == 17
    assert sum(r['cycles'] for r in result['state_counters']) == 640
    assert sum(r['input_accepted_bytes'] for r in result['state_counters']) == result['totals']['input_accepted_bytes']
    write_report(result, tmp_path/'tables')
    assert (tmp_path/'tables/layers.csv').exists()
    assert 'not network latency' in (tmp_path/'tables/report.md').read_text()


def test_host_timings_are_separate_and_prefix_not_counted_twice(downsample_evidence, tmp_path):
    from fpga.openeye.export_network_measurements import collect, write_report
    path = downsample_evidence / 'layer2-0-conv1-stage-verification.json'
    stage = load_json(path)
    operation = dict(operation='relu_int8_clamp_cast', elements=128, calls=1, wall_ns=2000)
    stage['host_operations'] = dict(forward_path=[operation], oracle_path=[dict(operation, wall_ns=9000)])
    write(path, stage)
    report_path = downsample_evidence / 'network-verification.json'
    report = load_json(report_path)
    report['stages'] = [stage if s['layer'] == stage['layer'] else s for s in report['stages']]
    report['blocks'][-1]['host_operations'] = dict(forward_path=[operation])
    write(report_path, report)
    seal(downsample_evidence)
    result = collect(downsample_evidence, COMMIT)
    assert result['totals']['total_cycles'] == 640
    actual = next(r for r in result['host_operation_totals'] if r['phase'] == 'forward_path')
    reference = next(r for r in result['host_operation_totals'] if r['phase'] == 'oracle_path')
    assert (actual['calls'], actual['wall_ns']) == (2, 4000)
    assert (reference['calls'], reference['wall_ns']) == (1, 9000)
    assert result['host_timing_coverage']['measured_stage_sites'] == 1
    assert collect(downsample_evidence, COMMIT)['host_operation_totals'] == result['host_operation_totals']
    write_report(result, tmp_path / 'report')
    assert (tmp_path / 'report/host_operations.csv').exists()


@pytest.mark.parametrize('change', [dict(wall_ns=-1), dict(wall_ns=float('nan')),
    dict(wall_ns=True), dict(elements=0), dict(operation='private-payload')])
def test_host_timings_reject_invalid_scalar_records(change):
    from fpga.openeye.export_network_measurements import host_timing_rows
    record = dict(operation='relu_int8_clamp_cast', elements=128, calls=1, wall_ns=2000)
    with pytest.raises(ValueError, match='host operation timing'):
        host_timing_rows('layer2.0.conv1', dict(forward_path=[dict(record, **change)]), 'openeye-rtl')


def test_report_cumulative_prefix_is_counted_once(downsample_evidence):
    from fpga.openeye.export_network_measurements import collect
    root = downsample_evidence
    report = load_json(root/'network-verification.json')
    names = ['layer1.0.conv1','layer1.0.conv2']
    for stage in report['stages'][:2]:
        stage['backend'] = 'openeye-rtl'
        slug = stage['layer'].replace('.','-')
        write(root/(slug+'-stage-verification.json'),stage)
        source = root/(slug+'-fixture-manifest.json')
        layer_evidence(root/slug,64,64,load_json(source)['execution'],stage['checked_values'],hashlib.sha256(source.read_bytes()).hexdigest())
    (root/'network-verification.json').unlink()
    summary = dict(result='SEGMENT_PASS',full_network_complete=False,domino_commit=COMMIT,
                   openeye_commit=EXPECTED_OPENEYE_COMMIT,completed_rtl_stages=names)
    write(root/'network-segment-000-verification.json',summary)
    write(root/'network-segment-001-verification.json',summary)
    seal(root)
    result=collect(root,COMMIT)
    assert result['unique_rtl_junit']==16 and result['totals']['total_cycles']==160
    summary['completed_rtl_stages'] = names * 2
    write(root/'network-segment-001-verification.json',summary);seal(root)
    with pytest.raises(ValueError,match='Duplicate'): collect(root,COMMIT)


def test_report_rejects_tampering_and_wrong_execution(downsample_evidence):
    from fpga.openeye.export_network_measurements import collect
    with pytest.raises(ValueError,match='execution'): collect(downsample_evidence,'c'*40)
    path=downsample_evidence/'layer2-0-conv1/hardware_metrics.json'
    data=load_json(path);data['records'][0]['total_cycles']+=1;write(path,data)
    with pytest.raises(ValueError,match='checksum'): collect(downsample_evidence,COMMIT)
    seal(downsample_evidence)
    with pytest.raises(ValueError): collect(downsample_evidence,COMMIT)


def test_report_missing_evidence_is_explicit_failure(tmp_path):
    import subprocess, sys
    script=Path(__file__).resolve().parents[1]/'fpga/openeye/export_network_measurements.py'
    output=tmp_path/'report'
    done=subprocess.run([sys.executable,str(script),'--evidence',str(tmp_path/'missing'),
                         '--output',str(output),'--domino-commit',COMMIT],capture_output=True)
    assert done.returncode==1
    result=load_json(output/'measurements.json')
    assert result['result']=='EVIDENCE_INCOMPLETE_OR_INVALID' and 'totals' not in result


@pytest.mark.parametrize('mode',['fc-smoke','full-network'])
def test_complete_network_metadata_and_gap_gate(tmp_path,mode):
    root = network_evidence(tmp_path,mode)
    result = verify(root,mode,COMMIT)
    assert result['rtl_junit'] == (1000 if mode=='fc-smoke' else 3720)
    report = load_json(root / 'network-verification.json')
    report['gap']['forwarded_int8_sha256'] = 'c' * 64
    write(root / 'network-verification.json',report)
    seal(root)
    with pytest.raises(ValueError,match='GAP'): verify(root,mode,COMMIT)


def test_complete_downsample_metadata_gate(downsample_evidence):
    result = verify(downsample_evidence,'downsample-block',COMMIT)
    assert result['rtl_junit'] == 64 and result['focused_junit'] == 8


@pytest.mark.parametrize('change', ['missing','wrong','storage','leaks-to-fc','leaks-to-projection'])
def test_kernel3_storage_provenance_is_required(downsample_evidence,change):
    name = ('focused-3' if change == 'leaks-to-fc' else 'layer2-0-downsample-0'
            if change == 'leaks-to-projection' else 'layer2-0-conv1')
    root = downsample_evidence / name
    path = root / 'regression_manifest.txt'
    lines = path.read_text().splitlines()
    if change == 'missing':
        lines = [line for line in lines if not line.startswith('runtime_kernel3_iact_patch_sha256=')]
    elif change != 'storage':
        lines = ['runtime_kernel3_iact_patch_sha256=' + 'c' * 64
                 if line.startswith('runtime_kernel3_iact_patch_sha256=') else line for line in lines]
    else:
        report = load_json(root / 'spatial_layer_verification.json')
        report['kernel3_activation_storage'] = 'default'
        write(root / 'spatial_layer_verification.json',report)
    path.write_text('\n'.join(lines) + '\n')
    with pytest.raises(ValueError,match='kernel3 activation'):
        verify_layer(root)


def test_full_geometry_downsample_metadata_gate(tmp_path):
    root = network_evidence(tmp_path, spatial_size=56)
    assert verify(root, 'downsample-block', COMMIT)['rtl_junit'] == 64
    assert load_json(root / 'network-verification.json')['output_shape'] == [128,28,28]


@pytest.mark.parametrize('declared_size', [8,56])
def test_crop_cannot_claim_full_downsample_geometry(downsample_evidence, declared_size):
    root = downsample_evidence
    report = load_json(root / 'network-verification.json')
    report['input_identity']['spatial_size'] = declared_size
    write(root / 'network-verification.json', report)
    seal(root)
    with pytest.raises(ValueError, match='geometry'):
        verify(root, 'downsample-block', COMMIT)


@pytest.mark.parametrize('change', ['missing', 'wrong', 'storage', 'leaks-to-kernel3'])
def test_projection_storage_provenance_is_required(downsample_evidence, change):
    name = 'layer2-0-conv1' if change == 'leaks-to-kernel3' else 'layer2-0-downsample-0'
    root = downsample_evidence / name
    path = root / 'regression_manifest.txt'
    lines = path.read_text().splitlines()
    if change == 'missing':
        lines = [line for line in lines if not line.startswith('runtime_projection_iact_patch_sha256=')]
    elif change in ('wrong', 'leaks-to-kernel3'):
        lines = ['runtime_projection_iact_patch_sha256=' + 'c' * 64
                 if line.startswith('runtime_projection_iact_patch_sha256=') else line for line in lines]
    else:
        report = load_json(root / 'spatial_layer_verification.json')
        report['projection_activation_storage'] = 'default'
        write(root / 'spatial_layer_verification.json', report)
    path.write_text('\n'.join(lines) + '\n')
    with pytest.raises(ValueError, match='projection activation'):
        verify_layer(root)


@pytest.mark.parametrize('change', ['pin','host','skip','order','source','missing-focused','cost','reference'])
def test_metadata_gate_rejects_incomplete_or_wrong_provenance(downsample_evidence,change):
    root = downsample_evidence
    report = load_json(root / 'network-verification.json')
    if change == 'pin': report['openeye_commit'] = 'c' * 40
    if change == 'host': report['stages'][-1]['backend'] = 'host-reference'
    if change == 'skip': report['stages'][-1]['input_sha256'] = 'c' * 64
    if change == 'order': report['stages'].reverse()
    if change == 'cost': report['eligible_for_network_cost'] = True
    if change == 'reference': report['result'] = 'HOST_REFERENCE_ONLY'
    if change == 'source':
        p = root / 'layer2-0-conv1-fixture-manifest.json'
        source = load_json(p)
        source['source_weight_shape'] = [64,64,3,3]
        write(p,source)
    if change == 'missing-focused':
        (root / 'focused-0/spatial-tiles/junit-0000.xml').unlink()
    write(root / 'network-verification.json',report)
    seal(root)  # Re-sealing metadata must not bypass structural checks.
    with pytest.raises(ValueError): verify(root,'downsample-block',COMMIT)


@pytest.mark.parametrize('change',['uncovered','checksum','private','symlink','alias','junit','counter','tile'])
def test_gate_rejects_bad_files_and_raw_execution(downsample_evidence,change):
    root = downsample_evidence
    if change == 'uncovered': (root / 'uncovered.txt').write_text('extra')
    if change == 'checksum': (root / 'network-verification.json').write_text('{}')
    if change == 'private':
        (root / 'activation.bin').write_bytes(b'private')
        seal(root)
    if change == 'symlink': (root / 'linked').symlink_to(root / 'network-verification.json')
    if change == 'alias':
        first = (root / 'SHA256SUMS').read_text().splitlines()[0]
        with (root / 'SHA256SUMS').open('a') as stream:
            digest, path = first.split('  ',1)
            stream.write(digest + '  ./' + path + '\n')
    if change == 'junit':
        (root / 'focused-0/spatial-tiles/junit-0000.xml').write_text('<testsuite><testcase><skipped/></testcase></testsuite>')
        seal(root)
    if change == 'counter':
        (root / 'focused-0/wrapper-conv-packed-layer.log').write_text('no accepted-handshake metrics')
        seal(root)
    if change == 'tile':
        write(root / 'focused-0/spatial-tiles/tile-0000.json',{})
        seal(root)
    with pytest.raises(ValueError): verify(root,'downsample-block',COMMIT)


def test_empty_junit_is_not_pass(tmp_path):
    layer_evidence(tmp_path,64,8)
    (tmp_path / 'spatial-tiles/junit-0000.xml').write_text('<testsuite tests="1" failures="0"/>')
    with pytest.raises(ValueError,match='JUnit'): verify_layer(tmp_path)


def test_manual_workflow_cannot_start_network_without_both_gates():
    path = SOURCE_ROOT.parents[1] / '.github/workflows/openeye-full-layer.yml'
    workflow = yaml.safe_load(path.read_text())
    assert set(workflow.get('on',workflow.get(True))) == {'workflow_dispatch'}
    gate, network = workflow['jobs']['residual'], workflow['jobs']['network_segment_0']
    assert 'downsample-block-pilot' in gate['strategy']['matrix']['gate']
    assert 'network-fc-smoke' in gate['strategy']['matrix']['gate']
    assert network['needs'] == 'residual' and 'always()' not in network['if']
    steps = network['steps']
    verify_index = next(i for i,s in enumerate(steps) if s['name'].startswith('Reverify complete same-commit gates'))
    execute_index = next(i for i,s in enumerate(steps) if s['name'].startswith('Execute or resume'))
    assert verify_index < execute_index
    assert '--mode downsample-block' in steps[verify_index]['run']
    assert '--mode fc-smoke' in steps[verify_index]['run']


@pytest.mark.parametrize('change', ['missing', 'wrong'])
def test_activation_packing_patch_identity_is_required(downsample_evidence, change):
    root = downsample_evidence / 'layer2-0-conv1'
    path = root / 'regression_manifest.txt'
    lines = path.read_text().splitlines()
    key = 'activation_packing_patch_sha256='
    lines = [line for line in lines if not line.startswith(key)]
    if change == 'wrong':
        lines.append(key + 'c' * 64)
    path.write_text('\n'.join(lines) + '\n')
    seal(downsample_evidence)
    with pytest.raises(ValueError, match='activation_packing_patch_sha256'):
        verify(downsample_evidence, 'downsample-block', COMMIT)
