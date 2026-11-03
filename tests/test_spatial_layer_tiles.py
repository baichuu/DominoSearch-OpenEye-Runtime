import copy
import json

import pytest

from fpga.openeye.spatial_layer_tiles import (
    axis_tiles, check_and_record_tile, finish_tiles, global_expected, tile_plan,
    validate_schedule, validate_weight_regions, record_tile_junit,
)


def execution(size=224, filters=64):
    return dict(input_width=size, input_height=size, input_channels=4,
                output_channels=filters, kernel_width=7, kernel_height=7,
                stride=2, padding="same")


@pytest.mark.parametrize("size", [8, 16, 24, 28, 32, 56, 224])
def test_axis_complete_disjoint_cores_and_halo(size):
    covered = []
    for tile in axis_tiles(size):
        assert 0 < tile["input_size"] <= 56
        assert tile["input_start"] % 2 == 0
        for x in range(tile["output_start"], tile["output_stop"]):
            covered.append(x)
            needed = set(range(max(0, x * 2 - 3), min(size, x * 2 + 4)))
            stored = set(range(tile["input_start"], tile["input_start"] + tile["input_size"]))
            assert needed <= stored
    assert covered == list(range(size // 2))


def test_full_conv1_is_200_bounded_tiles():
    plan = tile_plan(execution())
    assert len(plan) == 200
    assert max(t["x"]["input_size"] for t in plan) == 56


@pytest.mark.parametrize("kwargs", [dict(input_channels=3), dict(stride=1),
                                     dict(kernel_width=3), dict(output_channels=128),
                                     dict(output_channels=0), dict(output_channels=12),
                                     dict(padding="valid")])
def test_unsupported_paths_are_not_silently_tiled(kwargs):
    with pytest.raises(ValueError):
        tile_plan({**execution(), **kwargs})


def test_global_reference_uses_border_and_asymmetric_axes():
    weights = [[[[1 + x + 7 * y for x in range(7)] for y in range(7)]]]
    assert global_expected(weights, 32, 32, 0, 0, 0) == sum(
        1 + x + 7 * y for x in range(3, 7) for y in range(3, 7))
    assert global_expected(weights, 32, 32, 3, 3, 0) == sum(range(1, 50))


def test_actual_core_checks_and_global_coverage(tmp_path):
    ex = execution(64, 32)
    weights = [[[[1 + c + f + x + 7 * y for x in range(7)] for y in range(7)]
                for f in range(32)] for c in range(4)]
    plan = tile_plan(ex)
    reports = []
    for tile in plan:
        # Build local reference independently by direct convolution of an
        # all-one tile; retained core must match source global convolution.
        w, h = tile["x"]["input_size"], tile["y"]["input_size"]
        selected = [channel[tile["filter_start"]:tile["filter_stop"]] for channel in weights]
        decoded = [[[global_expected(weights, w, h, x, y, f) for y in range(h // 2)]
                    for x in range(w // 2)] for f in range(tile["filter_start"], tile["filter_stop"])]
        path = tmp_path / f"tile-{tile['index']}.json"
        pattern = dict(weights=selected, spatial_tile=tile, global_execution=ex, spatial_report=str(path))
        check_and_record_tile(pattern, decoded)
        reports.append(json.loads(path.read_text()))
    assert finish_tiles(plan, reports, ex)["checked_values"] == 32 * 32 * 32
    with pytest.raises(AssertionError, match="Missing"):
        finish_tiles(plan, reports[:-1], ex)
    broken = copy.deepcopy(reports)
    broken[0]["checked_values"] -= 1
    with pytest.raises(AssertionError, match="count"):
        finish_tiles(plan, broken, ex)
    broken = copy.deepcopy(reports)
    broken[0]["output_sha256"] = "incorrect"
    with pytest.raises(AssertionError, match="failed"):
        finish_tiles(plan, broken, ex)
    # Wrong DUT core is rejected before a report can be emitted.
    decoded[0][tile["x"]["crop_start"]][tile["y"]["crop_start"]] += 1
    pattern["spatial_report"] = str(tmp_path / "invalid.json")
    with pytest.raises(AssertionError, match="Global convolution mismatch"):
        check_and_record_tile(pattern, decoded)
    assert not (tmp_path / "invalid.json").exists()


def test_schedule_rejects_full_conv1_overflow_and_inclusive_bank_limit():
    names = ("iact_buffer_words_per_write", "iact_write_limit_0", "iact_read_inc_2",
             "trans_cycles_iact", "trans_cycles_wght", "trans_cycles_psum",
             "psum_output_words", "fsm_psum_limit", "iact_converter_max_cycles")
    fields = dict.fromkeys(names, 1)
    validate_schedule(fields)
    with pytest.raises(ValueError, match="16-bit"):
        validate_schedule({**fields, "iact_buffer_words_per_write": 83356})
    with pytest.raises(ValueError, match="18-bit"):
        validate_schedule({**fields, "psum_output_words": 1204224})
    with pytest.raises(ValueError, match="8-bit"):
        validate_schedule({**fields, "iact_converter_max_cycles": 3220})
    with pytest.raises(ValueError, match="bank depth"):
        validate_schedule({**fields, "trans_cycles_psum": 12 * (1 << 14)})


@pytest.mark.parametrize("kernel,stride,filters,blocks", [(1, 2, 128, 16), (3, 1, 64, 8)])
def test_filter_blocks_keep_all_source_filters(kernel, stride, filters, blocks):
    ex = dict(input_width=56, input_height=56, input_channels=64,
              output_channels=filters, kernel_width=kernel, kernel_height=kernel,
              stride=stride, padding="same")
    plan = tile_plan(ex)
    assert len(plan) == blocks
    assert [f for t in plan for f in range(t["filter_start"], t["filter_stop"])] == list(range(filters))
    assert all(t["x"]["input_size"] == t["y"]["input_size"] == 56 for t in plan)


@pytest.mark.parametrize("kernel,stride", [(1, 2), (3, 1)])
def test_filter_blocks_check_independent_signed_convolution(tmp_path, kernel, stride):
    ex = dict(input_width=8, input_height=4, input_channels=4,
              output_channels=16, kernel_width=kernel, kernel_height=kernel,
              stride=stride, padding="same")
    weights = [[[[((-1) ** (c + f + y + x)) * (1 + c + f + x + kernel * y)
                  for x in range(kernel)] for y in range(kernel)]
                for f in range(16)] for c in range(4)]
    plan = tile_plan(ex)
    reports = []
    for tile in plan:
        selected = [c[tile["filter_start"]:tile["filter_stop"]] for c in weights]
        decoded = [[[sum(weights[c][f][ky][kx]
                         for c in range(4) for ky in range(kernel) for kx in range(kernel)
                         if 0 <= x * stride + kx - kernel // 2 < 8
                         and 0 <= y * stride + ky - kernel // 2 < 4)
                     for y in range(4 // stride)] for x in range(8 // stride)]
                   for f in range(tile["filter_start"], tile["filter_stop"])]
        path = tmp_path / f"tile-{tile['index']}.json"
        pattern = dict(weights=selected, spatial_tile=tile, global_execution=ex,
                       spatial_report=str(path))
        check_and_record_tile(pattern, decoded)
        reports.append(json.loads(path.read_text()))
    result = finish_tiles(plan, reports, ex)
    assert result["execution"] == "filter-blocks"
    assert result["checked_values"] == 16 * (8 // stride) * (4 // stride)
    with pytest.raises(AssertionError, match="Missing"):
        finish_tiles(plan, reports[:-1], ex)
    decoded[0][0][0] += 1
    pattern["spatial_report"] = str(tmp_path / "bad.json")
    with pytest.raises(AssertionError, match="Global convolution mismatch"):
        check_and_record_tile(pattern, decoded)
    assert not (tmp_path / "bad.json").exists()


@pytest.mark.parametrize("updates", [dict(input_channels=65), dict(output_channels=7),
                                      dict(input_width=57), dict(input_width=55),
                                      dict(padding="valid")])
def test_unsupported_filter_block_capacity_is_rejected(updates):
    ex = dict(input_width=56, input_height=56, input_channels=64, output_channels=128,
              kernel_width=1, kernel_height=1, stride=2, padding="same")
    with pytest.raises(ValueError, match="Unsupported bounded"):
        tile_plan({**ex, **updates})


@pytest.mark.parametrize("weights,psums", [(172, 42), (182, 12), (88, 21), (0, 0)])
def test_measured_unsupported_weight_regions_fail_preflight(weights, psums):
    with pytest.raises(ValueError, match="weight-region geometry"):
        validate_weight_regions(dict(used_wght_per_PE=weights, used_psum_per_PE=psums))


def test_bounded_weight_regions_fit_fixed_spads():
    for weights, psums in [(32, 8), (96, 8), (112, 8)]:
        validate_weight_regions(dict(used_wght_per_PE=weights, used_psum_per_PE=psums))
    with pytest.raises(ValueError, match="SPAD capacity"):
        validate_weight_regions(dict(used_wght_per_PE=272, used_psum_per_PE=16))


def test_tile_junit_copies_real_result_without_overwrite(tmp_path):
    source = tmp_path / "simulator.xml"
    source.write_text('<testsuites><testsuite><testcase name="real" /></testsuite></testsuites>')
    target = tmp_path / "junit-000.xml"
    record_tile_junit(source, target)
    assert source.read_bytes() == target.read_bytes()
    with pytest.raises(FileExistsError):
        record_tile_junit(source, target)


@pytest.mark.parametrize("cases", ['', '<testcase><failure /></testcase>',
                                    '<testcase><error /></testcase>',
                                    '<testcase><skipped /></testcase>',
                                    '<testcase /><testcase />'])
def test_tile_junit_rejects_empty_failed_or_skipped_result(tmp_path, cases):
    source = tmp_path / "simulator.xml"
    source.write_text(f'<testsuite>{cases}</testsuite>')
    target = tmp_path / "junit-000.xml"
    with pytest.raises(AssertionError, match="one passing"):
        record_tile_junit(source, target)
    assert not target.exists()
