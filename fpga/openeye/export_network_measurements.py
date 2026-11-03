#!/usr/bin/env python3
"""Export audited counters, coverage and limitations; never infer a speedup."""
import argparse
import csv
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fpga.openeye.openeye_evidence_bundle import parse_key_values, parse_sha256s, sha256
from fpga.openeye.prepare_packed_layer_fixture import DEFAULT_PLAN, load_json
from fpga.openeye.verify_network_evidence import verify, verify_layer, EXPECTED_OPENEYE_COMMIT

COUNTERS = ('total_cycles', 'load_cycles', 'compute_cycles', 'stall_cycles',
            'input_stall_cycles', 'output_stall_cycles', 'accepted_weight_bytes',
            'input_accepted_bytes', 'output_accepted_bytes')
GAPS = [
    'No same-input dense versus mixed-N:M hardware comparison: no measured speedup or traffic reduction claim.',
    'Summed partial RTL cycles exclude host work, simulator setup and checkpoint I/O; not network latency.',
    'Stall cycles overlap load/compute; do not add them again to total cycles.',
    'No INT8 dataset accuracy evaluation; FP32 pruning accuracy remains separate.',
    'No r3 post-route timing/resources, power, or board measurement; do not reuse v6 results.',
    'Host stem, BN/requantization, partial accumulation, residual/ReLU, GAP and FC bias remain host operations.',
]


def collect(root, commit):
    root = Path(root)
    if root.is_symlink() or any(p.is_symlink() for p in root.rglob('*')):
        raise ValueError('Symlink in evidence')
    sums = parse_sha256s(root / 'SHA256SUMS')
    files = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name != 'SHA256SUMS'}
    if {str(Path(k)) for k in sums} != files:
        raise ValueError('Incomplete checksum coverage')
    for name, digest in sums.items():
        if Path(name).is_absolute() or '..' in Path(name).parts or sha256(root / name) != digest:
            raise ValueError('Invalid evidence checksum')
    final = root / 'network-verification.json'
    segment_reports = sorted(root.glob('network-segment-???-verification.json'))
    report = load_json(final if final.exists() else segment_reports[-1])
    if report['domino_commit'] != commit or report['openeye_commit'] != EXPECTED_OPENEYE_COMMIT:
        raise ValueError('Wrong execution commit or OpenEye pin')
    if final.exists():
        verify(root, report['mode'], commit)
    elif report['result'] != 'SEGMENT_PASS' or report.get('full_network_complete') is not False:
        raise ValueError('Invalid partial segment report')
    source_plan = {r['layer']: r for r in load_json(DEFAULT_PLAN)['layers'] if r['layer'] != 'conv1'}
    expected = list(source_plan)
    ingress = load_json(root / 'ingress-verification.json')['candidates']['mac20']
    declared = report.get('rtl_layers', report.get('completed_rtl_stages', []))
    if len(declared) != len(set(declared)) or any(n not in expected for n in declared):
        raise ValueError('Duplicate or unknown RTL stage')
    if not final.exists() and declared != expected[:len(declared)]:
        raise ValueError('Segment is not a complete ordered prefix')
    rows, states = [], []
    for name in expected:
        row = dict(layer=name, status='NO_COMPLETE_RTL_EVIDENCE')
        if name not in declared:
            rows.append(row)
            continue
        slug = name.replace('.', '-')
        stage = load_json(root / (slug + '-stage-verification.json'))
        source_path = root / (slug + '-fixture-manifest.json')
        source = load_json(source_path)
        provenance = parse_key_values(root / slug / 'regression_manifest.txt')
        if (stage['layer'] != name or stage['backend'] != 'openeye-rtl'
                or source['source_layer'] != name or provenance['fixture_manifest_sha256'] != sha256(source_path)):
            raise ValueError('Wrong stage/source binding')
        if (source['source_checkpoint_sha256'] != ingress['checkpoint_sha256']
                or source['source_packed_manifest_sha256'] != ingress['packed_manifest_sha256']
                or source['source_weight_shape'] != source_plan[name]['weight_shape']
                or source['scheme'] != source_plan[name]['expected_scheme']):
            raise ValueError('Wrong immutable source or mixed-N:M identity')
        junit = verify_layer(root / slug, stage)
        records = load_json(root / slug / 'hardware_metrics.json')['records']
        spatial = load_json(root / slug / 'spatial_layer_verification.json')
        row.update(status='VERIFIED_RTL', scheme=source['scheme'], rtl_junit=junit,
                   logical_checked_values=stage['checked_values'],
                   raw_checked_values=(sum(t['checked_values'] for t in spatial['tiles'])
                                       if all('checked_values' in t for t in spatial['tiles']) else None),
                   input_shape='x'.join(map(str, stage['input_shape'])),
                   output_shape='x'.join(map(str, stage['accumulator_shape'])),
                   dense_weight_elements=math.prod(source['source_weight_shape']),
                   retained_values=source.get('retained_values'),
                   quantization_max_abs_error=source.get('quantization', {}).get('max_abs_error'),
                   quantization_mean_abs_error=source.get('quantization', {}).get('mean_abs_error'),
                   source_manifest_sha256=sha256(source_path))
        for counter in COUNTERS:
            if counter in ('input_accepted_bytes', 'output_accepted_bytes'):
                word_key = counter.replace('bytes', 'words')
                row[counter] = sum(r[word_key] * (r['dma_bitwidth'] // 8) for r in records)
            else:
                row[counter] = sum(r[counter] for r in records)
        row['load_fraction'] = row['load_cycles'] / row['total_cycles'] if row['total_cycles'] else None
        row['stall_fraction'] = row['stall_cycles'] / row['total_cycles'] if row['total_cycles'] else None
        for field in ('input_values','input_zero_values','forwarded_values','forwarded_zero_values',
                      'forwarded_at_positive_limit','preparation_reference_wall_seconds',
                      'wrapper_and_output_read_wall_seconds','checking_affine_wall_seconds'):
            row[field] = stage.get(field)
        for state in sorted({s for r in records for key in ('state_cycles','input_accepted_words_by_state') for s in r[key]}):
            states.append(dict(layer=name, state=state,
                cycles=sum(r['state_cycles'].get(state, 0) for r in records),
                input_accepted_bytes=sum(r['input_accepted_words_by_state'].get(state, 0) * (r['dma_bitwidth'] // 8) for r in records),
                input_accepted_words=sum(r['input_accepted_words_by_state'].get(state, 0) for r in records)))
        rows.append(row)
    measured = [r for r in rows if r['status'] == 'VERIFIED_RTL']
    return dict(schema_version=1, result='AUDITED_MEASUREMENTS', domino_commit=commit,
                exporter_sha256=sha256(Path(__file__)),
                openeye_commit=report['openeye_commit'], evidence_sha256=sha256(root / 'SHA256SUMS'),
                scope=report.get('mode', 'completed-network-prefix'),
                full_network_verified=final.exists() and report.get('mode') == 'full-network',
                expected_rtl_layers=len(expected), verified_rtl_layers=len(measured),
                unique_rtl_junit=sum(r['rtl_junit'] for r in measured),
                aggregation='one evidence root; cumulative prefix counted once; gates/focused controls excluded',
                totals={k:sum(r[k] for r in measured) for k in COUNTERS},
                layers=rows, state_counters=states, limitations=GAPS)


def write_report(result, output):
    output.mkdir(parents=True, exist_ok=False)
    (output / 'measurements.json').write_text(json.dumps(result, indent=2) + '\n')
    for filename, records in [('layers.csv', result.get('layers', [])), ('states.csv', result.get('state_counters', []))]:
        columns = list(dict.fromkeys(k for row in records for k in row))
        with (output / filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(records)
    lines = ['# OpenEye measurement audit', '', 'Status: ' + result['result'], '']
    if result['result'] == 'AUDITED_MEASUREMENTS':
        lines += [f"Verified RTL stages: {result['verified_rtl_layers']}/{result['expected_rtl_layers']}; unique partial JUnit: {result['unique_rtl_junit']}.",
                  '', '| Layer | Status | Scheme | Total cycles | Weight bytes | Input bytes | Output bytes |',
                  '| --- | --- | --- | ---: | ---: | ---: | ---: |']
        for r in result['layers']:
            lines.append('| ' + ' | '.join(str(r.get(k, '—')) for k in ('layer','status','scheme','total_cycles','accepted_weight_bytes','input_accepted_bytes','output_accepted_bytes')) + ' |')
    else:
        lines += ['Evidence audit failed; no measurement totals certified.', '', result['error']]
    lines += ['', '## Limits', ''] + ['- ' + s for s in GAPS]
    (output / 'report.md').write_text('\n'.join(lines) + '\n')
    (output / 'SHA256SUMS').write_text(''.join(sha256(p) + '  ' + p.name + '\n' for p in sorted(output.iterdir()) if p.is_file()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--domino-commit', required=True)
    args = p.parse_args()
    if args.output.resolve().is_relative_to(args.evidence.resolve()):
        p.error('Report must be outside immutable/checkpoint evidence')
    try:
        result = collect(args.evidence, args.domino_commit)
    except (ValueError, KeyError, OSError, IndexError, TypeError, ET.ParseError) as exc:
        # Do not copy arbitrary exception strings/provider paths into public evidence.
        result = dict(result='EVIDENCE_INCOMPLETE_OR_INVALID', error=type(exc).__name__)
    write_report(result, args.output)
    if result['result'] != 'AUDITED_MEASUREMENTS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
