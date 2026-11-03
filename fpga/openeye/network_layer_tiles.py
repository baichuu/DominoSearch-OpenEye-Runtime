"""Opt-in bounded channel/filter execution for verified hybrid network inputs.

Every partial sum is checked against an independent integer oracle. No source
weight/mask is altered; host adds channel partial sums before BN. Not latency.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

try:
    from domino_runtime_activations import activation_record, convolution, dense, validate_activation
    from domino_spatial_tiles import record_tile_junit, validate_schedule, validate_weight_regions
except ImportError:
    from fpga.openeye.runtime_activations import activation_record, convolution, dense, validate_activation
    from fpga.openeye.spatial_layer_tiles import record_tile_junit, validate_schedule, validate_weight_regions


def channel_filter_plan(channels, filters):
    if channels <= 0 or channels % 64 or filters <= 0 or filters % 8:
        raise ValueError("Network tiling requires whole 64-channel and 8-filter slices")
    return [dict(index=i, channel_start=c, channel_stop=c + 64,
                 filter_start=f, filter_stop=f + 8)
            for i, (c, f) in enumerate((c, f) for c in range(0, channels, 64)
                                       for f in range(0, filters, 8))]


def convolution_execution(a, weights, execution):
    """Keep the native kernel and source weights; lower only spatial stride."""
    if ((execution['kernel_width'], execution['kernel_height']) not in ((1, 1), (3, 3))
            or execution['stride'] not in (1, 2) or max(a.shape[1:]) > 56):
        raise ValueError('Network runtime supports bounded kernel1/kernel3, stride1/2 only')
    padded = np.pad(a, ((0, 0), (0, a.shape[1] % 2), (0, a.shape[2] % 2)))
    return padded, weights, execution['kernel_width'], 1


def validate_retained_activation_capacity(used, capacity):
    """Retaining zeros needs every slot; the pinned count/tag fields are 4-bit."""
    if not isinstance(used, int) or not 0 < used <= min(capacity, 15):
        raise ValueError('Retained activations exceed PE SPAD or 4-bit count capacity')


def assemble(plan, reports, partials, shape):
    if not (len(plan) == len(reports) == len(partials)):
        raise AssertionError("Incomplete network partial/JUnit coverage")
    expected_channels = max(t['channel_stop'] for t in plan)
    if plan != channel_filter_plan(expected_channels, shape[0]):
        raise AssertionError("Channel/filter plan has gaps, duplicates or reordering")
    output = np.zeros(shape, dtype=np.int64)
    for tile, report, partial in zip(plan, reports, partials):
        if report['tile'] != tile or report['result'] != 'PASS' or report['output_sha256'] != report['reference_sha256']:
            raise AssertionError("Invalid network tile evidence")
        raw = np.asarray(partial, dtype='<i4').tobytes()
        if hashlib.sha256(raw).hexdigest() != report['output_sha256']:
            raise AssertionError("Partial output checksum mismatch")
        output[tile['filter_start']:tile['filter_stop']] += partial
    if output.min() < -(1 << 31) or output.max() >= (1 << 31):
        raise ValueError("Combined channel accumulator exceeds INT32")
    return output


def run_tiles(test_module, fixture, fixture_path, evidence):
    linear = fixture['layer_type'] == 'linear'
    execution = fixture['execution']
    shape = ([execution['input_features'], 1, 1] if linear else
             [execution['input_channels'], execution['input_width'], execution['input_height']])
    runtime = json.loads(Path(os.environ['OPENEYE_RUNTIME_ACTIVATIONS']).read_text())
    a = validate_activation(runtime, shape)
    weights = np.asarray(fixture['weights'], dtype=np.int64)
    channels, filters = (weights.shape[1], weights.shape[0]) if linear else weights.shape[:2]
    plan = channel_filter_plan(channels, filters)
    if not linear:
        # Zero-pad ONLY the far x/y border to an even shape. Output core is
        # cropped back after accumulation; no extra value enters a real halo.
        reference = convolution(validate_activation(runtime, shape), weights, execution['stride'])
        # The direct small stride-2 mapper produces zero outputs on this pin.
        # Execute the equivalent stride-1 correlation then retain even positions.
        # Keep projections native 1x1: center embedding introduces all-zero
        # kernel rows and fails with real zero-rich MAC20 activations. Neither
        # native stride-1 path changes source weights/masks or relaxes checks.
        a, weights, dut_kernel, dut_stride = convolution_execution(a, weights, execution)
    else:
        reference = dense(a[:, 0, 0], weights)
    evidence.mkdir()
    private_output = fixture_path.parent / 'network-output'
    private_output.mkdir()
    original_run = test_module.cocotb_test.simulator.run
    original_creator = test_module.vh_file_creator.create_vh_file_from_envvars
    original_generator = test_module.generator.create_regmap_params_vh_file
    identity = {}
    reports, partials = [], []

    def cached_run(**kwargs):
        if not linear:
            from open_eye.open_eye_parameters import OpenEyeParameters
            from open_eye.layer_parameters import LayerParameters
            layer = SimpleNamespace(name='conv2d', filters=8,
                input=SimpleNamespace(shape=(1, a.shape[1], a.shape[2], 64)),
                output=SimpleNamespace(shape=(1, a.shape[1], a.shape[2], 8)),
                kernel=SimpleNamespace(shape=(dut_kernel, dut_kernel, 64, 8)),
                kernel_size=(dut_kernel, dut_kernel), strides=(dut_stride, dut_stride))
            hardware = OpenEyeParameters(serial=True)
            topology = {"Previous_Layer": {layer.name: ["input_layer"]},
                        "Next_Layer": {layer.name: []}}
            parameters = LayerParameters({}, layer, hardware, topology, 0)
            validate_retained_activation_capacity(parameters.used_iact_per_PE, hardware.Iacts_per_PE)
            validate_weight_regions(vars(parameters))
            validate_schedule(vars(parameters), int(os.environ['BUFFER_WIDTH_PSUM']))
        sources = [Path(p) for p in kwargs['verilog_sources']]
        sources += list(Path(kwargs['sim_build']).glob('*.vh'))
        for parent in {p.parent for p in sources}:
            sources += list(parent.glob('*.vh'))
        current = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
        if identity and identity != current:
            raise AssertionError('HDL changed between network channel/filter slices')
        kwargs['extra_env']['COCOTB_TRACE'] = '0'
        if identity and kwargs.get('simulator') == 'verilator':
            from cocotb_test.simulator import Verilator
            class FrozenVerilator(Verilator):
                def build_command(self):
                    binary = Path(self.sim_dir) / self.toplevel_module
                    if not binary.is_file():
                        raise AssertionError('Missing frozen network binary')
                    return [[str(binary), *self.plus_args]]
            kwargs.pop('simulator')
            junit = FrozenVerilator(**kwargs).run()
        else:
            kwargs['force_compile'] = not bool(identity)
            junit = original_run(**kwargs)
        record_tile_junit(Path(junit), evidence / f'junit-{len(reports):04d}.xml')
        identity.update(current)

    try:
        test_module.cocotb_test.simulator.run = cached_run
        for tile in plan:
            if identity:
                test_module.vh_file_creator.create_vh_file_from_envvars = lambda *a, **k: None
                test_module.generator.create_regmap_params_vh_file = lambda *a, **k: None
            c, f = tile['channel_start'], tile['filter_start']
            selected = weights[f:f + 8, c:c + 64] if linear else weights[c:c + 64, f:f + 8]
            local_execution = (dict(input_features=64, output_features=8, dataflow='row_stationary') if linear else
                               {**execution, 'input_channels': 64, 'output_channels': 8,
                                'input_width': a.shape[1], 'input_height': a.shape[2],
                                'kernel_width': dut_kernel, 'kernel_height': dut_kernel,
                                'stride': dut_stride})
            pattern = {**fixture, 'weights': selected.tolist(), 'execution': local_execution,
                       'shape': list(selected.shape),
                       'runtime_tile': tile, 'runtime_activations': activation_record(a[c:c + 64], runtime['scale']),
                       'runtime_output_file': str(private_output / f"tile-{tile['index']:04d}.bin"),
                       'spatial_report': str(evidence / f"tile-{tile['index']:04d}.json")}
            if not linear:
                pattern['spatial_tile'] = {**tile, 'x': dict(input_start=0, crop_start=0),
                                          'y': dict(input_start=0, crop_start=0)}
            path = fixture_path.parent / 'network-runtime-pattern.json'
            path.write_text(json.dumps(pattern))
            os.environ['OPENEYE_WGHT_PATTERN_JSON'] = str(path)
            request = SimpleNamespace(node=SimpleNamespace(nodeid='domino_network_channel_tiles'))
            print(f"DOMINO_NETWORK_TILE {tile['index'] + 1}/{len(plan)} {tile}", flush=True)
            if linear:
                test_module.test_gemm_layer(INPUT_SIZE=64, OUTPUT_SIZE=8, CLUSTER_ROWS=3,
                    NUM_GLB_IACT=1, NUM_GLB_PSUM=4, NUM_GLB_WGHT=3,
                    DATAFLOW='row_stationary', PARALLEL_MACS=2, CLUSTER_COLUMNS=2, request=request)
            else:
                test_module.INPUT_SIZE_X = a.shape[1]
                test_module.NUM_FILTERS = 8
                test_module.KERNEL_SIZE_X = test_module.KERNEL_SIZE_Y = dut_kernel
                test_module.STRIDE = dut_stride
                test_module._run_conv_const('Convolution_Single', 1, 3, 1, 64, request,
                    ramp_iacts=False, psum_width=32, trans_words=4, bias_step=7,
                    input_height=a.shape[2], asymmetric_kernel=False)
            report = json.loads(Path(pattern['spatial_report']).read_text())
            reports.append(report)
            partial_shape = ((8,) if linear else (8, a.shape[1], a.shape[2]))
            partials.append(np.fromfile(pattern['runtime_output_file'], dtype='<i4').reshape(partial_shape))
        assembled_shape = ((filters,) if linear else (filters, *partials[0].shape[1:]))
        output = assemble(plan, reports, partials, assembled_shape)
        if not linear:
            output = output[:, :shape[1], :shape[2]][:, ::execution['stride'], ::execution['stride']]
        if not np.array_equal(output, reference):
            raise AssertionError('Combined full-source runtime output mismatch')
        raw = output.astype('<i4').tobytes()
        with Path(os.environ['OPENEYE_RUNTIME_OUTPUT']).open('xb') as stream:
            stream.write(raw)
        summary = dict(schema_version=1, rtl_correctness='PASS', execution='runtime-channel-filter-blocks',
                       activation_policy='verified runtime INT8; far-border zero padding',
                       activation_sha256=runtime['sha256'], output_sha256=hashlib.sha256(raw).hexdigest(),
                       reference_sha256=hashlib.sha256(raw).hexdigest(), checked_values=int(output.size),
                       channel_count=channels, filter_count=filters, tiles=reports,
                       logical_execution=execution,
                       execution_lowering='native Dense' if linear else f'native kernel{dut_kernel} stride1; optional decimation',
                       projection_activation_storage=('all positions including zeros' if not linear and dut_kernel == 1 else 'default'),
                       kernel3_activation_storage=('all positions including zeros' if not linear and dut_kernel == 3 else 'default'),
                       eligible_for_network_cost=False, board_ready=False)
        with (evidence.parent / 'spatial_layer_verification.json').open('x') as stream:
            json.dump(summary, stream, indent=2, sort_keys=True)
    finally:
        test_module.cocotb_test.simulator.run = original_run
        test_module.vh_file_creator.create_vh_file_from_envvars = original_creator
        test_module.generator.create_regmap_params_vh_file = original_generator
