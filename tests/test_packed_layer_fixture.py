import hashlib
import json
from pathlib import Path
import struct
import os
import subprocess

import pytest

from fpga.openeye.prepare_packed_layer_fixture import (
    DEFAULT_PLAN, linear_oi, load_json, prepare, selected_matrix,
    validate_wrapper_geometry, verify_fixture_bundle,
)


def _record(path: Path, dtype: str) -> dict:
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "dtype": dtype,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _source(
    root: Path,
    *,
    layer: str,
    layer_type: str,
    shape: list[int],
    grouped_order: str,
    n: int,
    masks: bytes | None,
    values: list[float],
) -> None:
    values_path = root / "values.bin"
    values_path.write_bytes(struct.pack(f"<{len(values)}f", *values))
    groups = 1
    for dimension in shape:
        groups *= dimension
    groups //= 4
    metadata = None
    if masks is not None:
        metadata_path = root / "metadata.bin"
        metadata_path.write_bytes(masks)
        metadata = _record(metadata_path, "uint8")
    manifest = {
        "checkpoint": {"sha256": "a" * 64},
        "scheme": {"sha256": "b" * 64},
        "layers": [
            {
                "index": 0,
                "name": layer,
                "module_path": layer,
                "type": layer_type,
                "n": n,
                "m": 4,
                "groups": groups,
                "value_count": len(values),
                "source_weight_shape": shape,
                "grouping": {"grouped_order": grouped_order},
                "values": _record(values_path, "float32"),
                "metadata": metadata,
            }
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _plan(path: Path, layer: dict) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model": "resnet18",
                "candidate": "test",
                "layers": [layer],
            }
        ),
        encoding="utf-8",
    )


def test_reconstructs_sparse_ohwi_convolution_as_iohw(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _source(
        source,
        layer="layer1.0.conv1",
        layer_type="conv2d",
        shape=[2, 4, 1, 1],
        grouped_order="OHWI",
        n=2,
        masks=bytes([0b0101, 0b1010]),
        values=[1.0, 3.0, 6.0, 8.0],
    )
    plan = tmp_path / "plan.json"
    _plan(
        plan,
        {
            "index": 0,
            "slug": "conv",
            "layer": "layer1.0.conv1",
            "type": "conv2d",
            "expected_scheme": "2:4",
            "expected_grouped_order": "OHWI",
            "weight_shape": [2, 4, 1, 1],
            "input_width": 8,
            "input_height": 4,
            "stride": 1,
        },
    )
    output = tmp_path / "fixture"

    manifest = prepare(source, "layer1.0.conv1", output, execution_plan=plan)
    payload = json.loads((output / "packed-layer-wrapper.json").read_text())

    assert manifest["scheme"] == "2:4"
    assert payload["shape"] == [4, 2, 1, 1]
    assert payload["weights"][0][0][0][0] != 0
    assert payload["weights"][1][0][0][0] == 0
    assert payload["weights"][2][0][0][0] != 0
    assert payload["weights"][3][0][0][0] == 0
    assert payload["weights"][0][1][0][0] == 0
    assert payload["weights"][1][1][0][0] != 0
    assert verify_fixture_bundle(output)["source_layer"] == "layer1.0.conv1"


def test_reconstructs_dense_oihw_convolution(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _source(
        source,
        layer="layer2.0.downsample.0",
        layer_type="conv2d",
        shape=[2, 4, 1, 1],
        grouped_order="OIHW",
        n=4,
        masks=None,
        values=[float(value) for value in range(1, 9)],
    )
    plan = tmp_path / "plan.json"
    _plan(
        plan,
        {
            "index": 0,
            "slug": "downsample",
            "layer": "layer2.0.downsample.0",
            "type": "conv2d",
            "expected_scheme": "4:4",
            "expected_grouped_order": "OIHW",
            "weight_shape": [2, 4, 1, 1],
            "input_width": 56,
            "input_height": 56,
            "stride": 2,
        },
    )
    output = tmp_path / "fixture"

    prepare(source, "layer2.0.downsample.0", output, execution_plan=plan)
    payload = json.loads((output / "packed-layer-wrapper.json").read_text())

    assert payload["shape"] == [4, 2, 1, 1]
    assert all(
        payload["weights"][input_][output_][0][0] != 0
        for input_ in range(4)
        for output_ in range(2)
    )


@pytest.mark.parametrize("kernel,n,ragged", [(1, 4, False), (3, 3, False), (3, 3, True)],
                         ids=["dense-k1", "uniform-k3", "ragged-k3"])
def test_prepare_filter_block_reproducer(tmp_path, kernel, n, ragged):
    """Small signed fixture; optional focused RTL, never a full MAC20 run."""
    source = tmp_path / "source"
    source.mkdir()
    channels = int(os.environ.get("OPENEYE_FILTER_BLOCK_CHANNELS", "4"))
    assert channels in (4, 64), "Focused fixture supports four or 64 input channels"
    width = int(os.environ.get("OPENEYE_FILTER_BLOCK_WIDTH", "32"))
    height = int(os.environ.get("OPENEYE_FILTER_BLOCK_HEIGHT", "8"))
    filters = int(os.environ.get("OPENEYE_FILTER_BLOCK_FILTERS", "16"))
    assert filters in (8, 16)
    groups = filters * channels * kernel * kernel // 4
    values = [float((-1) ** i * (1 + i % 23)) for i in range(groups * n)]
    masks = None
    if n == 3:
        # Same legal 3:4 scheme, but vary the zero location across OHWI groups.
        # Unlike uniform 0b1011 this creates unequal packed PE region lengths
        # and transport padding, exposing a next-tile word sent past the tail.
        masks = bytes((0b0111, 0b1011, 0b1101, 0b1110)[(i * 7 + i // 3) % 4]
                      if ragged else 0b1011 for i in range(groups))
    _source(source, layer="filter_block_smoke", layer_type="conv2d",
            shape=[filters, channels, kernel, kernel], grouped_order="OHWI", n=n,
            masks=masks, values=values)
    plan = tmp_path / "plan.json"
    stride = 2 if kernel == 1 else 1
    _plan(plan, dict(index=0, slug="filter-block-smoke", layer="filter_block_smoke",
                    type="conv2d", expected_scheme=f"{n}:4", expected_grouped_order="OHWI",
                    weight_shape=[filters, channels, kernel, kernel], input_width=width,
                    input_height=height, stride=stride))
    output = tmp_path / "fixture"
    prepare(source, "filter_block_smoke", output, execution_plan=plan,
            architecture="full-layer-r3")
    assert verify_fixture_bundle(output)["retained_values"] == groups * n
    if os.environ.get("OPENEYE_RUN_FILTER_BLOCK_SMOKE") == "1":
        runtime_env = {}
        if os.environ.get("OPENEYE_RUN_RUNTIME_ACTIVATION_SMOKE") == "1":
            import numpy as np
            from fpga.openeye.runtime_activations import activation_record
            a = np.fromfunction(lambda c, x, y: (c * 13 + x * 5 + y * 7) % 128,
                                (channels, width, height), dtype=int)
            activation_path = tmp_path / "runtime-input.json"
            activation_path.write_text(json.dumps(activation_record(a, 0.03125)))
            runtime_env = {"OPENEYE_RUNTIME_ACTIVATIONS": str(activation_path),
                           "OPENEYE_RUNTIME_OUTPUT": str(tmp_path / "runtime-output.bin")}
        root = Path(__file__).resolve().parents[1]
        openeye = Path(os.environ.get("OPENEYE_ROOT", root.parent / "OpenEye"))
        subprocess.run([str(root / "fpga/openeye/run_wrapper_conv_regression.sh"),
                        str(openeye), str(tmp_path / "rtl"), "packed-layer", str(output)],
                       env={**os.environ, **runtime_env, "OPENEYE_ARCHITECTURE": "full-layer-r3"}, check=True)
        result = json.loads((tmp_path / "rtl/spatial_layer_verification.json").read_text())
        assert result["rtl_correctness"] == "PASS"
        assert len(result["tiles"]) == filters // 8
        assert result["checked_values"] == filters * (width // stride) * (height // stride)
        if runtime_env:
            from fpga.openeye.runtime_activations import convolution
            w = json.loads((output / "packed-layer-wrapper.json").read_text())["weights"]
            actual = np.fromfile(tmp_path / "runtime-output.bin", dtype='<i4').reshape(
                filters, width // stride, height // stride)
            assert np.array_equal(actual, convolution(a, w, stride))
            metrics = json.loads((tmp_path / "rtl/hardware_metrics.json").read_text())
            assert metrics["validation"]["result"] == "PASS"
            assert not metrics["eligible_for_cost_table"]
            provenance = (tmp_path / "rtl/regression_manifest.txt").read_text()
            assert "activation_payload=verified-runtime-int8-cxy\n" in provenance
            assert "activation_payload=constant-one\n" not in provenance


@pytest.mark.parametrize('kind,kernel,stride,channels,size', [('conv2d', 3, 2, 128, 8), ('conv2d', 1, 2, 64, 8), ('conv2d', 3, 1, 64, 7), ('linear', 1, 1, 128, 1), ('conv2d', 1, 2, 64, 16), ('conv2d', 3, 2, 64, 16)])
def test_network_channel_tiles_smoke(tmp_path, kind, kernel, stride, channels, size):
    if os.environ.get('OPENEYE_RUN_NETWORK_TILE_SMOKE') != '1':
        pytest.skip('Explicit focused network RTL smoke only')
    import numpy as np
    from fpga.openeye.runtime_activations import activation_record, convolution, dense
    source = tmp_path / 'source'
    source.mkdir()
    shape = [8, channels] if kind == 'linear' else [8, channels, kernel, kernel]
    count = int(np.prod(shape))
    values = [float(1 + (i * 7 % 13)) * (-1 if i % 3 else 1) for i in range(count)]
    order = 'OI' if kind == 'linear' else 'OHWI'
    _source(source, layer='network_smoke', layer_type=kind, shape=shape,
            grouped_order=order, n=4, masks=None, values=values)
    row = dict(index=0, layer='network_smoke', type=kind, expected_scheme='4:4',
               expected_grouped_order=order, weight_shape=shape)
    row.update(dict(input_features=channels, output_features=8) if kind == 'linear' else
               dict(input_width=size, input_height=size, stride=stride))
    plan = tmp_path / 'plan.json'
    _plan(plan, row)
    fixture = tmp_path / 'fixture'
    prepare(source, 'network_smoke', fixture, execution_plan=plan, architecture='full-layer-r3')
    a_shape = (channels, 1, 1) if kind == 'linear' else (channels, size, size)
    a = np.fromfunction(lambda c,x,y: (13*c + 5*x + 7*y) % 128, a_shape, dtype=int)
    if kind == 'conv2d' and kernel == 1 and size == 16:
        # Empty channel groups at odd AND retained even positions. The old
        # center embedding adds zero kernel rows; the runtime must not hide
        # its mismatches by checking only decimated outputs.
        a[4:8,3,10] = 0
        a[16:20,4,8] = 0
        a[32:48,6,6] = 0
        a[16:20,15,1:11] = 0
    if kind == 'conv2d' and kernel == 3 and size == 16:
        # Whole zero stencils plus empty per-channel kernel rows, including
        # border/odd raw positions. Still compare every raw stride-1 output.
        a[:,4:7,4:7] = 0
        a[16:20,8:11,7:10] = 0
        a[0:4,0,:] = 0
        a[32:36,-1,:] = 0
    input_path = tmp_path / 'input.json'
    input_path.write_text(json.dumps(activation_record(a, .03125)))
    root = Path(__file__).resolve().parents[1]
    openeye = Path(os.environ.get('OPENEYE_ROOT', root.parent / 'OpenEye'))
    output = tmp_path / 'output.bin'
    subprocess.run([str(root / 'fpga/openeye/run_wrapper_conv_regression.sh'), str(openeye),
                    str(tmp_path / 'rtl'), 'packed-layer', str(fixture)], check=True,
                   env={**os.environ, 'OPENEYE_ARCHITECTURE':'full-layer-r3',
                        'OPENEYE_RUNTIME_CHANNEL_TILES':'1', 'OPENEYE_RUNTIME_ACTIVATIONS':str(input_path),
                        'OPENEYE_RUNTIME_OUTPUT':str(output)})
    weights = load_json(fixture / 'packed-layer-wrapper.json')['weights']
    ref = dense(a[:,0,0], weights) if kind == 'linear' else convolution(a, weights, stride)
    actual = np.fromfile(output, dtype='<i4').reshape(ref.shape)
    assert np.array_equal(actual, ref)
    report = load_json(tmp_path / 'rtl/spatial_layer_verification.json')
    assert len(report['tiles']) == channels // 64
    assert report['checked_values'] == ref.size
    assert not load_json(tmp_path / 'rtl/hardware_metrics.json')['eligible_for_cost_table']
    from fpga.openeye.verify_network_evidence import verify_layer
    assert verify_layer(tmp_path / 'rtl') == channels // 64


def test_reconstructs_linear_in_openeye_dram_oi_order(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _source(
        source,
        layer="fc",
        layer_type="linear",
        shape=[2, 4],
        grouped_order="OI",
        n=4,
        masks=None,
        values=[float(value) for value in range(1, 9)],
    )
    plan = tmp_path / "plan.json"
    _plan(
        plan,
        {
            "index": 0,
            "slug": "fc",
            "layer": "fc",
            "type": "linear",
            "expected_scheme": "4:4",
            "expected_grouped_order": "OI",
            "weight_shape": [2, 4],
            "input_features": 4,
            "output_features": 2,
        },
    )
    output = tmp_path / "fixture"

    prepare(source, "fc", output, execution_plan=plan)
    payload = json.loads((output / "packed-layer-wrapper.json").read_text())

    assert payload["shape"] == [2, 4]
    assert payload["layout"] == "OI"
    assert payload["weights"] == [[16, 32, 48, 64], [79, 95, 111, 127]]
    assert verify_fixture_bundle(output)["source_layer"] == "fc"

    # Even a checksum-consistent old IO fixture must be rejected, not reused.
    payload["layout"] = "IO"
    payload["weights"] = [list(column) for column in zip(*payload["weights"])]
    payload["shape"] = [4, 2]
    path = output / "packed-layer-wrapper.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    manifest = json.loads((output / "manifest.json").read_text())
    manifest["fixture_shape"] = [4, 2]
    manifest["file"] = {key: value for key, value in _record(path, "unused").items()
                        if key != "dtype"}
    (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="DRAM layout"):
        verify_fixture_bundle(output)


def test_dense_oi_reference_with_nonconstant_inputs():
    weights = linear_oi([1, 2, 3, 4, 5, 6, 7, 8], [2, 4], "OI")
    inputs = [2, -3, 5, -7]
    assert [sum(w * x for w, x in zip(row, inputs)) for row in weights] == [-17, -29]


def test_fixed_v6_rejects_kernel7_without_clamping_or_skipping():
    with pytest.raises(ValueError, match="requires 3 cluster rows"):
        validate_wrapper_geometry("conv2d", {"kernel_width": 7, "kernel_height": 7})
    with pytest.raises(ValueError, match="NOT passed"):
        selected_matrix(load_json(DEFAULT_PLAN))


def test_rows3_preflights_all_21_layers_on_one_architecture():
    plan = load_json(DEFAULT_PLAN)
    matrix = selected_matrix(plan, architecture="full-layer-r3")
    assert len(matrix["include"]) == 21
    assert {row["architecture"] for row in matrix["include"]} == {"full-layer-r3"}
    assert {row["architecture_id"] for row in matrix["include"]} == {
        "openeye-fpga-full-layer-r3-staging-depths"}
    assert [row["scheme"] for row in matrix["include"]] == [
        row["expected_scheme"] for row in plan["layers"]]
    with pytest.raises(ValueError, match="requires 4 cluster rows"):
        validate_wrapper_geometry("conv2d", {"kernel_width": 11, "kernel_height": 11},
                                  "full-layer-r3")


def test_unknown_architecture_is_rejected_even_for_dense():
    with pytest.raises(ValueError, match="Unknown wrapper architecture"):
        validate_wrapper_geometry("linear", {}, "typo")


def test_rows3_channel_padding_preserves_source_and_rejects_nonzero_padding(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _source(source, layer="conv1", layer_type="conv2d", shape=[4, 3, 7, 7],
            grouped_order="OHWI", n=4, masks=None,
            values=[float(i % 31 + 1) for i in range(4*3*7*7)])
    plan = tmp_path / "plan.json"
    _plan(plan, {"index": 0, "slug": "conv1", "layer": "conv1", "type": "conv2d",
                 "expected_scheme": "4:4", "expected_grouped_order": "OHWI",
                 "weight_shape": [4, 3, 7, 7], "input_width": 16, "input_height": 16,
                 "stride": 2})
    output = tmp_path / "fixture"
    manifest = prepare(source, "conv1", output, execution_plan=plan,
                       architecture="full-layer-r3")
    assert manifest["source_weight_shape"] == [4, 3, 7, 7]
    assert manifest["fixture_shape"] == [4, 4, 7, 7]
    assert manifest["retained_values"] == 588
    assert manifest["scheme"] == "4:4"
    assert manifest["execution"]["input_channels"] == 4
    assert manifest["input_channel_padding"] == 1
    assert verify_fixture_bundle(output)["source_checkpoint_sha256"] == "a" * 64
    path = output / "packed-layer-wrapper.json"
    payload = json.loads(path.read_text())
    payload["weights"][-1][0][0][0] = 1
    path.write_text(json.dumps(payload))
    manifest["file"] = {key: value for key, value in _record(path, "unused").items()
                        if key != "dtype"}
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="only zero weights"):
        verify_fixture_bundle(output)


@pytest.mark.parametrize("width", [1, 3, 5, 6])
def test_geometry_capacity_gate_accepts_fitting_widths(width):
    validate_wrapper_geometry("conv2d", {"kernel_width": width, "kernel_height": width})


def test_matrix_can_select_one_supported_layer_but_not_unknown_layer():
    plan = load_json(DEFAULT_PLAN)
    matrix = selected_matrix(plan, "fc")
    assert len(matrix["include"]) == 1
    assert matrix["include"][0]["index"] == 20
    with pytest.raises(ValueError, match="exactly once"):
        selected_matrix(plan, "typo")


def test_matrix_selects_only_requested_representative_layers():
    plan = load_json(DEFAULT_PLAN)
    names = ["conv1", "layer1.0.conv2", "layer2.0.downsample.0"]
    matrix = selected_matrix(plan, ", ".join(names), "full-layer-r3")
    assert [row["layer"] for row in matrix["include"]] == names
    assert [row["scheme"] for row in matrix["include"]] == ["4:4", "3:4", "4:4"]
    assert {row["architecture"] for row in matrix["include"]} == {"full-layer-r3"}
    with pytest.raises(ValueError, match="requires 3 cluster rows"):
        selected_matrix(plan, ",".join(names), "v6")
    with pytest.raises(ValueError, match="exactly once"):
        selected_matrix(plan, "fc,typo", "full-layer-r3")


@pytest.mark.parametrize("selection", ["", "fc,", ",fc", "fc,fc", "all,fc"])
def test_matrix_rejects_ambiguous_subsets(selection):
    with pytest.raises(ValueError, match="unique, nonempty"):
        selected_matrix(load_json(DEFAULT_PLAN), selection, "full-layer-r3")


def test_rejects_execution_plan_scheme_mismatch(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _source(
        source,
        layer="fc",
        layer_type="linear",
        shape=[2, 4],
        grouped_order="OI",
        n=4,
        masks=None,
        values=[float(value) for value in range(1, 9)],
    )
    plan = tmp_path / "plan.json"
    _plan(
        plan,
        {
            "layer": "fc",
            "type": "linear",
            "expected_scheme": "3:4",
            "expected_grouped_order": "OI",
            "weight_shape": [2, 4],
            "input_features": 4,
            "output_features": 2,
        },
    )

    with pytest.raises(ValueError, match="scheme"):
        prepare(source, "fc", tmp_path / "fixture", execution_plan=plan)
