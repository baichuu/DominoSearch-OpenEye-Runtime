"""Bounded spatial/filter convolution execution; not a network runtime.

Reuse the fixed r3 accelerator and unchanged verified IOHW weight tensor.
Only constant-one activation fixtures are supported here. Every retained core
output is checked against an independent global-coordinate integer reference;
the existing full local DMA/reference assertions remain enabled as well.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import struct
from types import SimpleNamespace
import xml.etree.ElementTree as ET


def axis_tiles(input_size: int, kernel: int = 7, stride: int = 2,
               max_input: int = 56) -> list[dict]:
    if (kernel, stride) != (7, 2) or input_size <= 0 or input_size % stride:
        raise ValueError("Spatial conv1 tiling requires positive even input, kernel7, stride2")
    halo = (kernel // 2 + stride - 1) // stride
    core_size = max_input // stride - 2 * halo
    if core_size <= 0 or max_input % stride:
        raise ValueError("Tile capacity must leave space for core and both halos")
    output_size = input_size // stride
    tiles = []
    for start in range(0, output_size, core_size):
        stop = min(start + core_size, output_size)
        origin = max(0, start - halo)
        end = min(output_size, stop + halo)
        tiles.append({"input_start": origin * stride, "input_size": (end - origin) * stride,
                      "output_start": start, "output_stop": stop,
                      "output_origin": origin, "crop_start": start - origin})
    return tiles


def tile_plan(execution: dict) -> list[dict]:
    geometry = (execution["kernel_width"], execution["kernel_height"], execution["stride"])
    if geometry in ((1, 1, 2), (3, 3, 1)):
        stride = execution["stride"]
        if (execution["padding"] != "same"
                or not 0 < execution["input_channels"] <= 64
                or not 0 < execution["output_channels"] <= 128
                or execution["output_channels"] % 8
                or any(not 0 < execution[axis] <= 56 or execution[axis] % stride
                       for axis in ("input_width", "input_height"))):
            raise ValueError("Unsupported bounded filter-only convolution")
        # A full 128-filter 1x1 schedule has unequal weight/PSUM region widths;
        # a full 64-filter 3x3 schedule exceeds r3's fixed PSUM bank depth.
        # Run complete eight-filter slices, with unchanged input/kernel/weights.
        x = dict(input_start=0, input_size=execution["input_width"],
                 output_start=0, output_stop=execution["input_width"] // stride,
                 output_origin=0, crop_start=0)
        y = dict(input_start=0, input_size=execution["input_height"],
                 output_start=0, output_stop=execution["input_height"] // stride,
                 output_origin=0, crop_start=0)
        return [dict(index=f // 8, x=x.copy(), y=y.copy(),
                     filter_start=f, filter_stop=f + 8)
                for f in range(0, execution["output_channels"], 8)]
    if (execution["kernel_width"], execution["kernel_height"], execution["stride"]) != (7, 7, 2):
        raise ValueError("Only conv1 kernel7 stride2 is supported by this execution path")
    if execution["padding"] != "same" or execution["output_channels"] > 64:
        raise ValueError("Unsupported padding or filter capacity for spatial conv1")
    if execution["input_channels"] != 4:
        raise ValueError("Spatial conv1 requires the verified 3-to-4 padded channel fixture")
    if execution["output_channels"] <= 0 or execution["output_channels"] % 8:
        raise ValueError("Spatial conv1 filter count must be positive and divisible by 8")
    # Kernel7 needs 14 regions/PE. The upstream 64-filter schedule disagrees
    # on region width (13 weight entries versus 12 PSUM entries). Execute
    # complete 8-filter blocks instead: 8 PSUMs/PE, exactly 112 weight entries.
    # A 16-filter pass reaches a separate r3 re-arm hang on the second filter
    # computation. Eight avoids that unsupported multi-filter pass entirely.
    # This also fits the unchanged 12-bit PSUM banks at spatial size <=56.
    return [{"index": index, "x": x, "y": y, "filter_start": f, "filter_stop": f + 8}
            for index, (x, y, f) in enumerate(
                (x, y, f) for x in axis_tiles(execution["input_width"])
                for y in axis_tiles(execution["input_height"])
                for f in range(0, execution["output_channels"], 8))]


def global_expected(weights: list, width: int, height: int, x: int, y: int,
                    filter_index: int, stride: int = 2) -> int:
    """Independent source convolution: INT8 ones, zero padding, no learned bias.

    OpenEye DRAM is [input][filter][kernel_y][kernel_x]; do not use mapper,
    local tile padding or its reference to decide global border behavior.
    """
    total = 0
    kernel_h = len(weights[0][filter_index])
    kernel_w = len(weights[0][filter_index][0])
    for channel in weights:
        for ky, row in enumerate(channel[filter_index]):
            for kx, value in enumerate(row):
                if (0 <= x * stride + kx - kernel_w // 2 < width
                        and 0 <= y * stride + ky - kernel_h // 2 < height):
                    total += value
    return total


def check_and_record_tile(pattern: dict, decoded: list) -> None:
    if "runtime_activations" in pattern:
        from domino_runtime_activations import check_runtime_tile
        check_runtime_tile(pattern, decoded)
        return
    tile = pattern["spatial_tile"]
    execution = pattern["global_execution"]
    actual_hash = hashlib.sha256()
    expected_hash = hashlib.sha256()
    count = 0
    # Interior points share the same complete-kernel sum. Cache the small
    # number of distinct global border positions without relying on DUT data.
    reference_cache = {}
    stride = execution["stride"]
    padx, pady = execution["kernel_width"] // 2, execution["kernel_height"] // 2
    for f in range(tile["filter_start"], tile["filter_stop"]):
        local_filter = f - tile["filter_start"]
        for x in range(tile["x"]["output_start"], tile["x"]["output_stop"]):
            for y in range(tile["y"]["output_start"], tile["y"]["output_stop"]):
                lx, ly = x - tile["x"]["output_origin"], y - tile["y"]["output_origin"]
                bx = min(x * stride, padx)
                by = min(y * stride, pady)
                ex = min(execution["input_width"] - x * stride, padx + 1)
                ey = min(execution["input_height"] - y * stride, pady + 1)
                key = (f, bx, by, ex, ey)
                if key not in reference_cache:
                    reference_cache[key] = global_expected(pattern["weights"],
                        execution["input_width"], execution["input_height"], x, y, local_filter,
                        stride)
                expected = reference_cache[key]
                actual = int(decoded[local_filter][lx][ly])
                if actual != expected:
                    raise AssertionError(f"Global convolution mismatch tile={tile['index']} "
                                         f"f={f}, x={x}, y={y}: DUT={actual}, ref={expected}")
                actual_hash.update(struct.pack("<i", actual))
                expected_hash.update(struct.pack("<i", expected))
                count += 1
    report = {"tile": tile, "checked_values": count, "result": "PASS",
              "output_sha256": actual_hash.hexdigest(), "reference_sha256": expected_hash.hexdigest()}
    with Path(pattern["spatial_report"]).open("x", encoding="utf-8") as stream:
        json.dump(report, stream, sort_keys=True)


def finish_tiles(plan: list[dict], reports: list[dict], execution: dict) -> dict:
    if len(reports) != len(plan) or any(r["tile"] != t or r["result"] != "PASS"
                                       or r["output_sha256"] != r["reference_sha256"]
                                       for t, r in zip(plan, reports)):
        raise AssertionError("Missing, reordered, or failed spatial tile evidence")
    coverage = set()
    expected_total = 0
    for tile, report in zip(plan, reports):
        points = {(f, x, y) for f in range(tile["filter_start"], tile["filter_stop"])
                  for x in range(tile["x"]["output_start"], tile["x"]["output_stop"])
                  for y in range(tile["y"]["output_start"], tile["y"]["output_stop"])}
        if coverage.intersection(points):
            raise AssertionError("Spatial cores overlap")
        coverage.update(points)
        count = len(points)
        if report["checked_values"] != count:
            raise AssertionError("Wrong tile core output count")
        expected_total += count
    expected_points = {(f, x, y) for f in range(execution["output_channels"])
                       for x in range(execution["input_width"] // execution["stride"])
                       for y in range(execution["input_height"] // execution["stride"])}
    if coverage != expected_points:
        raise AssertionError("Spatial cores do not cover the complete global layer")
    return {"schema_version": 1, "rtl_correctness": "PASS",
            "execution": "spatial-halo-tiles" if execution["kernel_width"] == 7 else "filter-blocks",
            "activation_policy": "INT8 constant one; global zero padding",
            "bias_policy": "zero; source convolution has no learned bias",
            "global_execution": execution, "tiles": reports, "checked_values": expected_total,
            "measurement_level": "rtl-wrapper-simulation", "board_ready": False}


def validate_schedule(fields: dict, psum_bits: int = 12) -> None:
    """Reject wraparound rather than letting the DMA register packer truncate.

    Checked widths come from the maintained pinned regmap/RTL. Bias words
    stripe across 6 clusters x 2 PSUM banks; the address count must not wrap.
    """
    widths = {"iact_buffer_words_per_write": 16, "iact_write_limit_0": 16,
              "iact_read_inc_2": 16, "trans_cycles_iact": 16,
              "trans_cycles_wght": 16, "trans_cycles_psum": 18,
              "psum_output_words": 18, "fsm_psum_limit": 17,
              "iact_converter_max_cycles": 8}
    for name, width in widths.items():
        value = fields[name]
        if value != int(value) or not 0 <= value < 1 << width:
            raise ValueError(f"{name}={value} exceeds unsigned {width}-bit r3 capacity")
    # psum_cnt stores the depth itself, so equality to 2**bits also wraps.
    if fields["trans_cycles_psum"] // 12 >= 1 << psum_bits:
        raise ValueError("Bias/PSUM bank depth would wrap; tile this layer")


def validate_weight_regions(fields: dict) -> None:
    """Reject the measured ceil/floor mismatch before reference/RTL execution."""
    logical = int(fields["used_wght_per_PE"])
    region = int(fields["used_psum_per_PE"])
    if not 0 < region <= 32 or logical <= 0 or logical % region:
        raise ValueError(f"Unsupported wrapper weight-region geometry: weights={logical}, psums={region}")
    if logical // region > 16 or logical > 192:
        raise ValueError("Weight regions exceed fixed PE SPAD capacity")


def record_tile_junit(source: Path, destination: Path) -> None:
    """Keep real simulator JUnit, not a synthesized pass certificate."""
    data = source.read_bytes()
    cases = list(ET.fromstring(data).iter("testcase"))
    if len(cases) != 1 or any(case.find(tag) is not None for case in cases
                              for tag in ("failure", "error", "skipped")):
        raise AssertionError("Tile simulator JUnit must contain exactly one passing testcase")
    with destination.open("xb") as stream:
        stream.write(data)


def run_tiles(test_module, fixture: dict, fixture_path: Path, evidence: Path) -> None:
    """Run each tile independently using the maintained upstream helper.

    Geometry is runtime configuration, not HDL parameters. Freeze one build
    directory and generated headers for this process; compile once, then reuse
    only that unchanged HDL/architecture. No cross-run build cache is used.
    """
    execution = fixture["execution"]
    if os.environ.get("OPENEYE_ZERO_BIAS") != "1":
        raise ValueError("Packed convolution must retain the source zero-bias policy")
    plan = tile_plan(execution)
    runtime = None
    if os.environ.get("OPENEYE_RUNTIME_ACTIVATIONS"):
        from domino_runtime_activations import validate_activation
        runtime = json.loads(Path(os.environ["OPENEYE_RUNTIME_ACTIVATIONS"]).read_text())
        validate_activation(runtime, [execution["input_channels"], execution["input_width"],
                                     execution["input_height"]])
        if execution["kernel_width"] != 3 or execution["stride"] != 1:
            raise ValueError("Runtime pilot is limited to kernel3 stride1 filter blocks")
        private_output = fixture_path.parent / "runtime-output"
        private_output.mkdir()
    evidence.mkdir()
    reports = []
    original_creator = test_module.vh_file_creator.create_vh_file_from_envvars
    original_generator = test_module.generator.create_regmap_params_vh_file
    original_run = test_module.cocotb_test.simulator.run
    build_sources = {}

    def cached_run(**kwargs):
        from open_eye.open_eye_parameters import OpenEyeParameters
        from open_eye.layer_parameters import LayerParameters
        env = kwargs["extra_env"]
        width, height = int(env["INPUT_SIZE_X"]), int(env["INPUT_SIZE_Y"])
        channels = int(env["INPUT_CHANNELS"])
        kernel_w, kernel_h = int(env["KERNEL_SIZE_X"]), int(env["KERNEL_SIZE_Y"])
        stride = int(env["STRIDE"])
        layer = SimpleNamespace(name="conv2d", filters=8,
            input=SimpleNamespace(shape=(1, width, height, channels)),
            output=SimpleNamespace(shape=(1, width // stride, height // stride, 8)),
            kernel=SimpleNamespace(shape=(kernel_w, kernel_h, channels, 8)),
            kernel_size=(kernel_w, kernel_h), strides=(stride, stride))
        topology = {"Previous_Layer": {layer.name: ["input_layer"]},
                    "Next_Layer": {layer.name: []}}
        parameters = LayerParameters({}, layer, OpenEyeParameters(serial=True), topology, 0)
        validate_weight_regions(vars(parameters))
        validate_schedule(vars(parameters), int(os.environ["BUFFER_WIDTH_PSUM"]))
        sources = [Path(p) for p in kwargs["verilog_sources"]]
        sources += list(Path(kwargs["sim_build"]).glob("*.vh"))
        # Included parameter headers can live beside the HDL, not only build.
        for parent in {p.parent for p in sources}:
            sources += list(parent.glob("*.vh"))
        identity = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
        if build_sources and identity != build_sources:
            raise AssertionError("HDL or generated parameters changed between spatial tiles")
        if build_sources and kwargs.get("simulator") == "verilator":
            # cocotb-test 0.2.6's Verilator ignores force_compile=False and
            # unconditionally emits compile/make commands. Use its normal
            # environment/JUnit runner with a binary-only command after the
            # identical source/header checksum gate above.
            from cocotb_test.simulator import Verilator

            class FrozenVerilator(Verilator):
                def build_command(self):
                    binary = Path(self.sim_dir) / self.toplevel_module
                    if not binary.is_file():
                        raise AssertionError("Missing frozen spatial Verilator binary")
                    return [[str(binary), *self.plus_args]]

            kwargs.pop("simulator")
            results_xml = FrozenVerilator(**kwargs).run()
        else:
            kwargs["force_compile"] = not bool(build_sources)
            results_xml = original_run(**kwargs)
        record_tile_junit(Path(results_xml), evidence / f"junit-{len(reports):03d}.xml")
        build_sources.update(identity)

    try:
        test_module.cocotb_test.simulator.run = cached_run
        for tile in plan:
            if build_sources:
                test_module.vh_file_creator.create_vh_file_from_envvars = lambda *a, **k: None
                test_module.generator.create_regmap_params_vh_file = lambda *a, **k: None
            pattern = {**fixture, "spatial_tile": tile, "global_execution": execution,
                       "execution": {**execution, "input_width": tile["x"]["input_size"],
                                     "input_height": tile["y"]["input_size"], "output_channels": 8},
                       "shape": [execution["input_channels"], 8,
                                 execution["kernel_height"], execution["kernel_width"]],
                       "weights": [channel[tile["filter_start"]:tile["filter_stop"]]
                                   for channel in fixture["weights"]],
                       "spatial_report": str(evidence / f"tile-{tile['index']:03d}.json")}
            tile_path = fixture_path.parent / "spatial-runtime-pattern.json"
            if runtime is not None:
                pattern["runtime_activations"] = runtime
                pattern["runtime_output_file"] = str(private_output / f"tile-{tile['index']:03d}.bin")
            tile_path.write_text(json.dumps(pattern), encoding="utf-8")
            os.environ["OPENEYE_WGHT_PATTERN_JSON"] = str(tile_path)
            test_module.INPUT_SIZE_X = tile["x"]["input_size"]
            test_module.INPUT_SIZE_Y = tile["y"]["input_size"]
            test_module.NUM_FILTERS = 8
            test_module.KERNEL_SIZE_X = execution["kernel_width"]
            test_module.KERNEL_SIZE_Y = execution["kernel_height"]
            test_module.STRIDE = execution["stride"]
            print(f"DOMINO_SPATIAL_TILE {tile['index'] + 1}/{len(plan)} {tile}", flush=True)
            test_module._run_conv_const("Convolution_Single", 1, 3, 1, execution["input_channels"],
                SimpleNamespace(node=SimpleNamespace(nodeid="domino_conv_filter_tiles")),
                ramp_iacts=False, psum_width=32, trans_words=4, bias_step=7,
                input_height=tile["y"]["input_size"], asymmetric_kernel=False)
            reports.append(json.loads(Path(pattern["spatial_report"]).read_text()))
        result = finish_tiles(plan, reports, execution)
        if runtime is not None:
            import numpy as np
            output = np.empty((execution["output_channels"], execution["input_width"],
                               execution["input_height"]), dtype='<i4')
            for tile, report in zip(plan, reports):
                raw = (private_output / f"tile-{tile['index']:03d}.bin").read_bytes()
                if hashlib.sha256(raw).hexdigest() != report["output_sha256"]:
                    raise AssertionError("Private runtime output checksum mismatch")
                output[tile["filter_start"]:tile["filter_stop"]] = np.frombuffer(raw, dtype='<i4').reshape(
                    8, execution["input_width"], execution["input_height"])
            with Path(os.environ["OPENEYE_RUNTIME_OUTPUT"]).open('xb') as stream:
                stream.write(output.tobytes())
            result["activation_policy"] = "verified nonuniform INT8 runtime tensor; zero padding"
            result["activation_sha256"] = runtime["sha256"]
        with (evidence.parent / "spatial_layer_verification.json").open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
    finally:
        test_module.vh_file_creator.create_vh_file_from_envvars = original_creator
        test_module.generator.create_regmap_params_vh_file = original_generator
        test_module.cocotb_test.simulator.run = original_run
