#!/usr/bin/env python3
"""Build one OpenEye INT8 wrapper fixture from a packed MAC20 layer.

This tool reconstructs the selected N:M stream already stored in a verified
packed-v1 artifact.  It does not invent the missing dense weights and it does
not change the layer's selected scheme.  Convolution fixtures use OpenEye's
IOHW list order; the fully connected fixture uses OpenEye DRAM's OI order
(not the IO order of a Keras Dense kernel).

The resulting payload is for independent RTL layer correctness and traffic
measurement.  It is not calibrated end-to-end INT8 inference evidence.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Any


FIXTURE_FILENAME = "packed-layer-wrapper.json"
DEFAULT_PLAN = Path(__file__).with_name("resnet18_mac20_execution.json")
WRAPPER_ARCHITECTURES = {
    "v6": {"architecture_id": "openeye-fpga-first-layer-v6-staging-depths", "cluster_rows": 2},
    "full-layer-r3": {"architecture_id": "openeye-fpga-full-layer-r3-staging-depths", "cluster_rows": 3},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def file_record(path: Path) -> dict[str, Any]:
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def verify_source_file(root: Path, record: dict[str, Any], dtype: str) -> Path:
    filename = record.get("file")
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError("Packed source contains an unsafe filename.")
    path = root / filename
    if (
        record.get("dtype") != dtype
        or not path.is_file()
        or path.stat().st_size != int(record.get("bytes", -1))
        or sha256(path) != record.get("sha256")
    ):
        raise ValueError(f"Packed source identity mismatch: {path}")
    return path


def read_float32_le(path: Path) -> list[float]:
    data = path.read_bytes()
    if len(data) % 4:
        raise ValueError(f"FP32 file size is not divisible by four: {path}")
    values = [item[0] for item in struct.iter_unpack("<f", data)]
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"Packed layer contains a non-finite value: {path}")
    return values


def plan_layer(plan: dict[str, Any], layer_name: str) -> dict[str, Any]:
    matches = [row for row in plan.get("layers", []) if row.get("layer") == layer_name]
    if len(matches) != 1:
        raise ValueError(f"Execution plan must identify {layer_name!r} exactly once.")
    return matches[0]


def validate_wrapper_geometry(
    layer_type: str, execution: dict[str, Any], architecture: str = "v6",
) -> None:
    """Reject a known mapper capacity failure, not certify RTL correctness.

    Both configurations retain PEs_Y=3; Clusters_Y is explicitly selected.
    Upstream calculate_used_PEs uses kernel_size[0] (kernel width); then
    calculate_used_Y_cluster divides by floor(Clusters_Y / used_Y_cluster).
    Do not clamp that divisor or silently change the fixed architecture.
    """
    if architecture not in WRAPPER_ARCHITECTURES:
        raise ValueError(f"Unknown wrapper architecture: {architecture}")
    rows = WRAPPER_ARCHITECTURES[architecture]["cluster_rows"]
    if layer_type == "linear":
        return
    if layer_type != "conv2d":
        raise ValueError(f"Unsupported packed layer type: {layer_type}")
    width = int(execution["kernel_width"])
    height = int(execution["kernel_height"])
    if min(width, height) <= 0:
        raise ValueError("Kernel dimensions must be positive.")
    required_rows = (width + 2) // 3
    if required_rows > rows:
        raise ValueError(
            f"Unsupported {architecture} convolution kernel {height}x{width}: "
            f"requires {required_rows} cluster rows with PEs_Y=3, but Clusters_Y={rows}. "
            "The upstream mapper would divide by zero (Y_Cluster_Packages=0). "
            "Kernel decomposition needs separate implementation/validation; "
            "a larger architecture needs separate resource/timing evidence. "
            "This layer has NOT passed; no layer is silently skipped."
        )


def selected_matrix(
    plan: dict[str, Any], selection: str = "all", architecture: str = "v6",
) -> dict[str, Any]:
    """Preflight the requested matrix before any expensive simulator job starts."""
    selection = selection.strip()
    if selection == "all":
        layers = plan["layers"]
    else:
        names = [name.strip() for name in selection.split(",")]
        if not all(names) or "all" in names or len(set(names)) != len(names):
            raise ValueError("Select unique, nonempty layer names; 'all' must be used alone.")
        layers = [plan_layer(plan, name) for name in names]
    for row in layers:
        if row["type"] == "conv2d":
            validate_wrapper_geometry("conv2d", {
                "kernel_height": row["weight_shape"][2],
                "kernel_width": row["weight_shape"][3],
            }, architecture)
        else:
            validate_wrapper_geometry(row["type"], {}, architecture)
    return {"include": [
        {"index": row["index"], "slug": row["slug"], "layer": row["layer"],
         "layer_type": row["type"], "scheme": row["expected_scheme"],
         "architecture": architecture,
         "architecture_id": WRAPPER_ARCHITECTURES[architecture]["architecture_id"]}
        for row in layers
    ]}


def packed_layer(manifest: dict[str, Any], layer_name: str) -> dict[str, Any]:
    matches = [
        row
        for row in manifest.get("layers", [])
        if row.get("name") == layer_name or row.get("module_path") == layer_name
    ]
    if len(matches) != 1:
        raise ValueError(f"Packed manifest must identify {layer_name!r} exactly once.")
    return matches[0]


def validate_plan_identity(row: dict[str, Any], planned: dict[str, Any]) -> None:
    scheme = f"{row.get('n')}:{row.get('m')}"
    checks = {
        "type": (row.get("type"), planned.get("type")),
        "scheme": (scheme, planned.get("expected_scheme")),
        "grouped order": (
            row.get("grouping", {}).get("grouped_order"),
            planned.get("expected_grouped_order"),
        ),
        "weight shape": (row.get("source_weight_shape"), planned.get("weight_shape")),
    }
    mismatches = [
        f"{name}: packed={actual!r}, plan={expected!r}"
        for name, (actual, expected) in checks.items()
        if actual != expected
    ]
    if mismatches:
        raise ValueError("Packed layer does not match execution plan: " + "; ".join(mismatches))


def decode_and_quantize(
    root: Path, row: dict[str, Any]
) -> tuple[list[int], dict[str, Any], dict[str, Any]]:
    n = int(row["n"])
    m = int(row["m"])
    groups = int(row["groups"])
    if m != 4 or n not in {2, 3, 4}:
        raise ValueError(f"Unsupported selected scheme {n}:{m}; expected 2/3/4:4.")

    values_path = verify_source_file(root, row["values"], "float32")
    values = read_float32_le(values_path)
    if len(values) != groups * n or len(values) != int(row["value_count"]):
        raise ValueError("Packed FP32 value count is inconsistent with N:M metadata.")
    max_abs = max((abs(value) for value in values), default=0.0)
    if max_abs == 0:
        raise ValueError("Cannot quantize an all-zero packed layer.")
    scale = max_abs / 127.0

    metadata_path: Path | None = None
    if n == m:
        if row.get("metadata") is not None:
            raise ValueError("A 4:4 packed layer must not carry sparse metadata.")
        masks = None
    else:
        metadata = row.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"Selected {n}:{m} layer has no bitmask metadata.")
        metadata_path = verify_source_file(root, metadata, "uint8")
        masks = metadata_path.read_bytes()
        if len(masks) != groups:
            raise ValueError("Packed layer must have one N:M mask per group.")

    dense = [0] * (groups * m)
    value_cursor = 0
    forced_nonzero = 0
    source_zero_values = 0
    absolute_error_sum = 0.0
    max_abs_error = 0.0
    quantized_min = 127
    quantized_max = -127
    for group in range(groups):
        mask = (1 << m) - 1 if masks is None else masks[group]
        if mask >> m or mask.bit_count() != n:
            raise ValueError(f"Group {group} has invalid {n}:{m} mask 0x{mask:02x}.")
        for position in range(m):
            if not mask & (1 << position):
                continue
            value = values[value_cursor]
            value_cursor += 1
            if value == 0:
                source_zero_values += 1
                raise ValueError(
                    "A retained packed value is exactly zero; zero-skipping cannot preserve "
                    "the declared selected scheme."
                )
            qvalue = max(-127, min(127, int(round(value / scale))))
            if qvalue == 0:
                qvalue = 1 if value > 0 else -1
                forced_nonzero += 1
            dense[group * m + position] = qvalue
            error = abs(value - qvalue * scale)
            absolute_error_sum += error
            max_abs_error = max(max_abs_error, error)
            quantized_min = min(quantized_min, qvalue)
            quantized_max = max(quantized_max, qvalue)
    if value_cursor != len(values):
        raise AssertionError("Packed layer reconstruction did not consume every value.")

    source_identity: dict[str, Any] = {
        "values": {**file_record(values_path), "dtype": "float32"},
        "metadata": None,
    }
    if metadata_path is not None:
        source_identity["metadata"] = {
            **file_record(metadata_path),
            "dtype": "uint8",
        }
    quantization = {
        "scheme": "symmetric-per-tensor-selected-layer",
        "scale": scale,
        "zero_point": 0,
        "qmin": -127,
        "qmax": 127,
        "rounding": "round-to-nearest-ties-to-even",
        "retained_nonzero_policy": "round-to-zero values become sign(value) * 1",
        "source_max_abs": max_abs,
        "source_zero_values": source_zero_values,
        "forced_nonzero_values": forced_nonzero,
        "max_abs_error": max_abs_error,
        "mean_abs_error": absolute_error_sum / len(values),
        "quantized_min": quantized_min,
        "quantized_max": quantized_max,
    }
    return dense, quantization, source_identity


def conv_iohw(dense: list[int], shape: list[int], grouped_order: str) -> list[Any]:
    output_channels, input_channels, kernel_h, kernel_w = shape
    if len(dense) != output_channels * input_channels * kernel_h * kernel_w:
        raise ValueError("Decoded convolution payload does not match its OIHW shape.")

    def value(output: int, input_: int, y: int, x: int) -> int:
        if grouped_order == "OHWI":
            index = (((output * kernel_h + y) * kernel_w + x) * input_channels + input_)
        elif grouped_order == "OIHW":
            index = (((output * input_channels + input_) * kernel_h + y) * kernel_w + x)
        else:
            raise ValueError(f"Unsupported convolution grouped order: {grouped_order}")
        return dense[index]

    return [
        [
            [
                [value(output, input_, y, x) for x in range(kernel_w)]
                for y in range(kernel_h)
            ]
            for output in range(output_channels)
        ]
        for input_ in range(input_channels)
    ]


def linear_oi(dense: list[int], shape: list[int], grouped_order: str) -> list[Any]:
    output_features, input_features = shape
    if grouped_order != "OI":
        raise ValueError(f"Unsupported linear grouped order: {grouped_order}")
    if len(dense) != output_features * input_features:
        raise ValueError("Decoded linear payload does not match its OI shape.")
    return [
        dense[output * input_features:(output + 1) * input_features]
        for output in range(output_features)
    ]


def nested_shape(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    if not value:
        return [0]
    child_shapes = {tuple(nested_shape(item)) for item in value}
    if len(child_shapes) != 1:
        raise ValueError("Fixture has a ragged weight tensor.")
    return [len(value), *next(iter(child_shapes))]


def validate_int8(value: Any) -> int:
    if isinstance(value, list):
        return sum(validate_int8(item) for item in value)
    if isinstance(value, bool) or not isinstance(value, int) or not -128 <= value <= 127:
        raise ValueError("Fixture contains a non-INT8 weight value.")
    return int(value != 0)


def prepare(
    packed_root: Path,
    layer_name: str,
    output: Path,
    *,
    execution_plan: Path = DEFAULT_PLAN,
    architecture: str = "v6",
) -> dict[str, Any]:
    packed_root = packed_root.expanduser().resolve()
    output = output.expanduser().resolve()
    execution_plan = execution_plan.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {output}")

    packed_manifest_path = packed_root / "manifest.json"
    packed_manifest = load_json(packed_manifest_path)
    plan = load_json(execution_plan)
    if plan.get("schema_version") != 1:
        raise ValueError("Unsupported execution-plan schema.")
    planned = plan_layer(plan, layer_name)
    row = packed_layer(packed_manifest, layer_name)
    validate_plan_identity(row, planned)

    dense, quantization, source_identity = decode_and_quantize(packed_root, row)
    if architecture not in WRAPPER_ARCHITECTURES:
        raise ValueError(f"Unknown wrapper architecture: {architecture}")
    input_channel_padding = 0
    source_shape = list(row["source_weight_shape"])
    grouped_order = row["grouping"]["grouped_order"]
    if row["type"] == "conv2d":
        weights = conv_iohw(dense, source_shape, grouped_order)
        output_channels, input_channels, kernel_h, kernel_w = source_shape
        # Serial activation transport carries pairs. Preserve the source tensor
        # and retained positions; pad only the execution representation.
        if architecture == "full-layer-r3" and input_channels % 2:
            weights.append([[[0 for _ in range(kernel_w)] for _ in range(kernel_h)]
                            for _ in range(output_channels)])
            input_channels += 1
            input_channel_padding = 1
        fixture_shape = [input_channels, output_channels, kernel_h, kernel_w]
        layout = "IOHW"
        execution = {
            "input_width": int(planned["input_width"]),
            "input_height": int(planned["input_height"]),
            "input_channels": input_channels,
            "output_channels": output_channels,
            "kernel_height": kernel_h,
            "kernel_width": kernel_w,
            "stride": int(planned["stride"]),
            "padding": "same",
        }
    elif row["type"] == "linear":
        weights = linear_oi(dense, source_shape, grouped_order)
        output_features, input_features = source_shape
        fixture_shape = [output_features, input_features]
        layout = "OI"
        execution = {
            "input_features": int(planned["input_features"]),
            "output_features": int(planned["output_features"]),
            "dataflow": "row_stationary",
        }
        if execution != {
            "input_features": input_features,
            "output_features": output_features,
            "dataflow": "row_stationary",
        }:
            raise ValueError("Linear execution dimensions do not match the packed weight shape.")
    else:
        raise ValueError(f"Unsupported packed layer type: {row['type']}")

    retained_values = int(row["value_count"])
    if validate_int8(weights) != retained_values:
        raise AssertionError("Quantized fixture does not preserve every retained N:M position.")
    if nested_shape(weights) != fixture_shape:
        raise AssertionError("Generated fixture shape is inconsistent.")

    output.mkdir(parents=True)
    fixture = {
        "schema_version": 1,
        "source_layer": row["module_path"],
        "layer_type": row["type"],
        "scheme": f"{row['n']}:{row['m']}",
        "layout": layout,
        "shape": fixture_shape,
        "execution": execution,
        "input_channel_padding": input_channel_padding,
        "payload": "symmetric INT8 of the selected packed FP32 layer",
        "precision": "signed INT8; independent wrapper correctness only",
        "retained_values": retained_values,
        "quantization": quantization,
        "weights": weights,
    }
    fixture_path = output / FIXTURE_FILENAME
    with fixture_path.open("w", encoding="utf-8") as stream:
        json.dump(fixture, stream, indent=2, sort_keys=True)
        stream.write("\n")

    manifest = {
        "schema_version": 1,
        "method": "openeye-selected-packed-layer-int8-fixture",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "claim_level": "independent-weighted-layer-rtl-correctness-not-end-to-end-inference",
        "model": plan["model"],
        "candidate": plan["candidate"],
        "source_layer": row["module_path"],
        "source_layer_index": int(row["index"]),
        "layer_type": row["type"],
        "scheme": f"{row['n']}:{row['m']}",
        "grouped_order": grouped_order,
        "source_weight_shape": source_shape,
        "fixture_shape": fixture_shape,
        "execution": execution,
        "input_channel_padding": input_channel_padding,
        "retained_values": retained_values,
        "precision": "signed INT8 from selected packed FP32 values; no model accuracy claim",
        "quantization": quantization,
        "source_packed_manifest_sha256": sha256(packed_manifest_path),
        "source_execution_plan_sha256": sha256(execution_plan),
        "source_checkpoint_sha256": packed_manifest["checkpoint"]["sha256"],
        "source_scheme_sha256": packed_manifest["scheme"]["sha256"],
        "source_files": source_identity,
        "file": file_record(fixture_path),
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def verify_fixture_bundle(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve()
    manifest = load_json(root / "manifest.json")
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported packed-layer fixture manifest schema.")
    if manifest.get("method") != "openeye-selected-packed-layer-int8-fixture":
        raise ValueError("Unexpected packed-layer fixture method.")
    record = manifest.get("file")
    if not isinstance(record, dict) or record.get("file") != FIXTURE_FILENAME:
        raise ValueError("Packed-layer fixture manifest has an invalid file record.")
    fixture_path = root / FIXTURE_FILENAME
    if (
        not fixture_path.is_file()
        or fixture_path.stat().st_size != int(record.get("bytes", -1))
        or sha256(fixture_path) != record.get("sha256")
    ):
        raise ValueError(f"Packed-layer fixture identity mismatch: {fixture_path}")
    fixture = load_json(fixture_path)
    for key in ("source_layer", "layer_type", "scheme", "execution", "retained_values",
                "input_channel_padding"):
        if fixture.get(key) != manifest.get(key):
            raise ValueError(f"Packed-layer fixture metadata mismatch: {key}")
    if fixture.get("shape") != manifest.get("fixture_shape"):
        raise ValueError("Packed-layer fixture shape metadata mismatch.")
    source_shape = manifest["source_weight_shape"]
    if fixture["layer_type"] == "linear":
        expected_shape, expected_layout = source_shape, "OI"
    else:
        output_channels, input_channels, height, width = source_shape
        padding = fixture.get("input_channel_padding", 0)
        if padding not in (0, 1) or isinstance(padding, bool):
            raise ValueError("Invalid execution channel padding.")
        if padding:
            if input_channels % 2 != 1 or validate_int8(fixture["weights"][-1]) != 0:
                raise ValueError("Execution channel padding must contain only zero weights.")
            input_channels += 1
        expected_shape, expected_layout = [input_channels, output_channels, height, width], "IOHW"
    if fixture.get("layout") != expected_layout or fixture["shape"] != expected_shape:
        raise ValueError("Packed-layer fixture does not match OpenEye DRAM layout; regenerate it.")
    weights = fixture.get("weights")
    if nested_shape(weights) != fixture["shape"]:
        raise ValueError("Packed-layer fixture tensor shape mismatch.")
    if validate_int8(weights) != int(manifest["retained_values"]):
        raise ValueError("Packed-layer fixture retained-position count mismatch.")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packed-root", type=Path)
    parser.add_argument("--layer")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execution-plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--verify-root", type=Path)
    parser.add_argument("--check-wrapper-geometry", action="store_true",
                        help="Reject known unsupported geometry before simulation.")
    parser.add_argument("--architecture", choices=WRAPPER_ARCHITECTURES, default="v6")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.check_wrapper_geometry and args.verify_root is None:
        raise SystemExit("--check-wrapper-geometry requires --verify-root.")
    if args.verify_root is not None:
        if args.packed_root is not None or args.layer is not None or args.output is not None:
            raise SystemExit("--verify-root cannot be combined with fixture creation.")
        result = verify_fixture_bundle(args.verify_root)
        if args.check_wrapper_geometry:
            try:
                validate_wrapper_geometry(result["layer_type"], result["execution"], args.architecture)
            except ValueError as error:
                raise SystemExit(str(error)) from error
        print(
            f"Verified selected packed-layer fixture for {result['source_layer']} "
            f"({result['scheme']})."
        )
        return
    if args.packed_root is None or args.layer is None or args.output is None:
        raise SystemExit("--packed-root, --layer, and --output are required.")
    result = prepare(
        args.packed_root,
        args.layer,
        args.output,
        execution_plan=args.execution_plan,
        architecture=args.architecture,
    )
    print(
        f"Prepared {result['source_layer']} {result['scheme']} selected-layer fixture "
        f"at {args.output.expanduser().resolve()}."
    )


if __name__ == "__main__":
    main()
