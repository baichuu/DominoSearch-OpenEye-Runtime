#!/usr/bin/env python3
"""Create wrapper fixtures from first-layer mixed N:M candidates.

The default payloads are deterministic low-amplitude signed-INT8-safe position
tokens.  ``--payload-mode int8-symmetric`` instead quantizes the actual packed
FP32 candidate values with one scale shared by all three schemes.  Both modes
are wrapper-correctness experiments, not model-level INT8 accuracy results.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import struct


SCHEMES = ("4:4", "3:4", "2:4")
EXPECTED_SHAPE = [64, 64, 3, 3]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def verify_fixture_bundle(root: Path) -> dict:
    """Verify a generated wrapper fixture bundle by manifest identity."""
    root = root.expanduser().resolve()
    manifest_path = root / "manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported wrapper fixture manifest schema.")
    if manifest.get("source_layer") != "layer1.0.conv1":
        raise ValueError("Wrapper fixture does not identify layer1.0.conv1.")
    shape = manifest.get("fixture_shape_iohw")
    if not (
        isinstance(shape, list)
        and len(shape) == 4
        and all(isinstance(value, int) and value > 0 for value in shape)
    ):
        raise ValueError("Wrapper fixture has an invalid IOHW shape.")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(SCHEMES):
        raise ValueError("Wrapper fixture must contain exactly 4:4, 3:4, and 2:4.")
    for scheme, record in files.items():
        filename = record.get("file")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError(f"Unsafe wrapper fixture filename for {scheme}.")
        path = root / filename
        if (
            not path.is_file()
            or path.stat().st_size != int(record["bytes"])
            or sha256(path) != record["sha256"]
        ):
            raise ValueError(f"Wrapper fixture identity mismatch: {path}")
        payload = load_json(path)
        if payload.get("scheme") != scheme or payload.get("shape") != shape:
            raise ValueError(f"Wrapper fixture metadata mismatch: {path}")
        if int(payload.get("retained_values", -1)) != int(record["retained_values"]):
            raise ValueError(f"Wrapper fixture retained count mismatch: {path}")
    return manifest


def verify_file(root: Path, record: dict, *, dtype: str) -> Path:
    path = root / record["file"]
    if (
        record.get("dtype") != dtype
        or path.stat().st_size != int(record["bytes"])
        or sha256(path) != record["sha256"]
    ):
        raise ValueError(f"Candidate file identity mismatch: {path}")
    return path


def read_float32_le(path: Path) -> list[float]:
    data = path.read_bytes()
    if len(data) % 4:
        raise ValueError(f"FP32 file size is not divisible by four: {path}")
    values = [item[0] for item in struct.iter_unpack("<f", data)]
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"Non-finite candidate value: {path}")
    return values


def candidate_values_by_position(root: Path, candidate: dict) -> dict[int, float]:
    groups = int(candidate["groups"])
    n, m = int(candidate["n"]), int(candidate["m"])
    if m != 4 or n not in {2, 3, 4}:
        raise ValueError(f"Unsupported candidate {n}:{m}.")
    values_path = verify_file(root, candidate["values"], dtype="float32")
    values = read_float32_le(values_path)
    if int(candidate["value_count"]) != groups * n:
        raise ValueError("Candidate value count does not match N:M metadata.")
    if len(values) != groups * n:
        raise ValueError("Candidate FP32 payload count does not match N:M metadata.")

    metadata = candidate.get("metadata")
    if metadata is None:
        if n != m:
            raise ValueError("Sparse candidate has no metadata.")
        masks = bytes([0b1111]) * groups
    else:
        masks = verify_file(root, metadata, dtype="uint8").read_bytes()
        if len(masks) != groups:
            raise ValueError("Candidate metadata does not have one mask per group.")

    retained: dict[int, float] = {}
    cursor = 0
    for group, mask in enumerate(masks):
        positions = [position for position in range(m) if mask & (1 << position)]
        if len(positions) != n:
            raise ValueError(f"Group {group} has invalid {n}:{m} mask 0x{mask:02x}.")
        for position in positions:
            retained[group * m + position] = values[cursor]
            cursor += 1
    if cursor != len(values):
        raise AssertionError("Candidate reconstruction did not consume all FP32 values.")
    return retained


def retained_positions(root: Path, candidate: dict) -> set[int]:
    return set(candidate_values_by_position(root, candidate))


def quantize_symmetric_nonzero(
    values_by_position: dict[int, float], scale: float
) -> tuple[dict[int, int], dict]:
    """Quantize retained values while preserving the declared N:M mask.

    A retained nonzero FP32 value that rounds to zero is mapped to signed one.
    Otherwise OpenEye's zero-skipping transport would silently turn an N:M
    retained position into a pruned position and the hardware experiment would
    no longer represent the candidate named in the manifest.
    """
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("INT8 scale must be finite and positive.")
    quantized: dict[int, int] = {}
    forced_nonzero = 0
    errors = []
    for position, value in values_by_position.items():
        qvalue = int(round(value / scale))
        qvalue = max(-127, min(127, qvalue))
        if value != 0 and qvalue == 0:
            qvalue = 1 if value > 0 else -1
            forced_nonzero += 1
        quantized[position] = qvalue
        errors.append(abs(value - qvalue * scale))
    return quantized, {
        "forced_nonzero_values": forced_nonzero,
        "max_abs_error": max(errors, default=0.0),
        "mean_abs_error": math.fsum(errors) / len(errors) if errors else 0.0,
        "quantized_min": min(quantized.values(), default=0),
        "quantized_max": max(quantized.values(), default=0),
    }


def ohwi_position(
    output_channel: int,
    input_channel: int,
    kernel_y: int,
    kernel_x: int,
) -> int:
    _, input_channels, kernel_h, kernel_w = EXPECTED_SHAPE
    return (
        ((output_channel * kernel_h + kernel_y) * kernel_w + kernel_x)
        * input_channels
        + input_channel
    )


def build_fixture(
    retained: set[int],
    *,
    n: int,
    input_start: int,
    input_count: int,
    output_start: int,
    output_count: int,
    payload_by_position: dict[int, int] | None = None,
) -> tuple[list, int]:
    if input_start % 4 or input_count % 4:
        raise ValueError("Input-channel slice must contain complete OHWI M=4 groups.")
    output_limit, input_limit, kernel_h, kernel_w = EXPECTED_SHAPE
    if not (0 <= input_start < input_start + input_count <= input_limit):
        raise ValueError("Input-channel slice is outside the source tensor.")
    if not (0 <= output_start < output_start + output_count <= output_limit):
        raise ValueError("Output-channel slice is outside the source tensor.")

    weights = []
    retained_in_slice = 0
    for input_channel in range(input_start, input_start + input_count):
        input_weights = []
        for output_channel in range(output_start, output_start + output_count):
            kernel = []
            for kernel_y in range(kernel_h):
                row = []
                for kernel_x in range(kernel_w):
                    position = ohwi_position(
                        output_channel, input_channel, kernel_y, kernel_x
                    )
                    if position in retained:
                        value = (
                            payload_by_position[position]
                            if payload_by_position is not None
                            else position % 7 + 1
                        )
                        if value == 0:
                            raise AssertionError(
                                "A retained position became zero in the wrapper payload."
                            )
                        row.append(value)
                        retained_in_slice += 1
                    else:
                        row.append(0)
                kernel.append(row)
            input_weights.append(kernel)
        weights.append(input_weights)

    groups_in_slice = output_count * kernel_h * kernel_w * input_count // 4
    if retained_in_slice != groups_in_slice * n:
        raise AssertionError(
            f"Fixture retained {retained_in_slice} values; expected {groups_in_slice * n}."
        )
    return weights, retained_in_slice


def prepare(
    candidate_root: Path,
    output: Path,
    *,
    input_start: int = 0,
    input_count: int = 4,
    output_start: int = 0,
    output_count: int = 4,
    payload_mode: str = "token",
) -> dict:
    candidate_root = candidate_root.expanduser().resolve()
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {output}")
    source_manifest_path = candidate_root / "manifest.json"
    source_manifest = load_json(source_manifest_path)
    source = source_manifest.get("source", {})
    if (
        source.get("module_path") != "layer1.0.conv1"
        or source.get("weight_shape") != EXPECTED_SHAPE
        or source.get("grouping", {}).get("grouped_order") != "OHWI"
    ):
        raise ValueError("Expected the locked ResNet-18 layer1.0.conv1 OHWI candidate set.")
    if set(source_manifest.get("candidates", {})) != set(SCHEMES):
        raise ValueError("Candidate set must contain exactly 4:4, 3:4, and 2:4.")
    if payload_mode not in {"token", "int8-symmetric"}:
        raise ValueError(f"Unsupported payload mode: {payload_mode}")

    candidate_values = {
        scheme: candidate_values_by_position(
            candidate_root, source_manifest["candidates"][scheme]
        )
        for scheme in SCHEMES
    }
    quantization = None
    quantized_values: dict[str, dict[int, int]] = {}
    quantization_stats: dict[str, dict] = {}
    if payload_mode == "int8-symmetric":
        max_abs = max(abs(value) for value in candidate_values["4:4"].values())
        if max_abs == 0:
            raise ValueError("Cannot derive an INT8 scale from an all-zero source layer.")
        scale = max_abs / 127.0
        for scheme in SCHEMES:
            quantized_values[scheme], quantization_stats[scheme] = (
                quantize_symmetric_nonzero(candidate_values[scheme], scale)
            )
        quantization = {
            "scheme": "symmetric-per-tensor-shared-source-scale",
            "source": "4:4 candidate maximum absolute FP32 value",
            "scale": scale,
            "zero_point": 0,
            "qmin": -127,
            "qmax": 127,
            "rounding": "round-to-nearest-ties-to-even",
            "retained_nonzero_policy": "round-to-zero values become sign(value) * 1",
            "source_max_abs": max_abs,
        }

    output.mkdir(parents=True)
    files = {}
    for scheme in SCHEMES:
        candidate = source_manifest["candidates"][scheme]
        retained = set(candidate_values[scheme])
        if len(retained) != int(candidate["value_count"]):
            raise ValueError(f"Retained position count mismatch for {scheme}.")
        weights, retained_in_slice = build_fixture(
            retained,
            n=int(candidate["n"]),
            input_start=input_start,
            input_count=input_count,
            output_start=output_start,
            output_count=output_count,
            payload_by_position=quantized_values.get(scheme),
        )
        if payload_mode == "int8-symmetric":
            payload_description = "shared-scale symmetric INT8 of packed FP32 candidate values"
            precision = "signed INT8; actual candidate weights; wrapper correctness only"
        else:
            payload_description = "global OHWI position modulo 7 plus one; pruned positions are zero"
            precision = "representative signed-INT8-safe tokens; no checkpoint quantization"
        payload = {
            "schema_version": 1,
            "scheme": scheme,
            "layout": "IOHW",
            "shape": [input_count, output_count, 3, 3],
            "source_slice": {
                "input_channel_start": input_start,
                "output_channel_start": output_start,
            },
            "payload": payload_description,
            "precision": precision,
            "retained_values": retained_in_slice,
            "weights": weights,
        }
        if quantization is not None:
            payload["quantization"] = quantization
            payload["quantization_stats"] = quantization_stats[scheme]
        filename = f"candidate-{scheme.replace(':', 'x')}-wrapper.json"
        path = output / filename
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        files[scheme] = {
            "file": filename,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "retained_values": retained_in_slice,
            "density": int(candidate["n"]) / int(candidate["m"]),
        }
        if quantization is not None:
            files[scheme]["quantization_stats"] = quantization_stats[scheme]

    if payload_mode == "int8-symmetric":
        method = "openeye-wrapper-actual-candidate-int8-fixtures"
        claim_level = "wrapper-actual-candidate-int8-correctness"
        precision = "signed INT8 from actual packed FP32 values; no model accuracy claim"
    else:
        method = "openeye-wrapper-track-a-candidate-fixtures"
        claim_level = "wrapper-representative-payload-only"
        precision = "representative signed-INT8-safe tokens; no checkpoint quantization"
    manifest = {
        "schema_version": 1,
        "method": method,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "claim_level": claim_level,
        "source_candidate_manifest_sha256": sha256(source_manifest_path),
        "source_checkpoint_sha256": source["checkpoint_sha256"],
        "source_layer": source["module_path"],
        "source_weight_shape": EXPECTED_SHAPE,
        "fixture_shape_iohw": [input_count, output_count, 3, 3],
        "source_slice": {
            "input_channel_start": input_start,
            "output_channel_start": output_start,
        },
        "payload_mode": payload_mode,
        "precision": precision,
        "files": files,
    }
    if quantization is not None:
        manifest["quantization"] = quantization
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-root", type=Path)
    parser.add_argument("--input-start", type=int, default=0)
    parser.add_argument("--input-count", type=int, default=4)
    parser.add_argument("--output-start", type=int, default=0)
    parser.add_argument("--output-count", type=int, default=4)
    parser.add_argument(
        "--payload-mode", choices=("token", "int8-symmetric"), default="token"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.verify_root is not None:
        if args.candidate_root is not None or args.output is not None:
            raise SystemExit("--verify-root cannot be combined with fixture creation.")
        manifest = verify_fixture_bundle(args.verify_root)
        print(
            f"Verified {manifest.get('payload_mode', 'legacy-token')} wrapper fixtures: "
            f"{args.verify_root.expanduser().resolve()}"
        )
        return
    if args.candidate_root is None or args.output is None:
        raise SystemExit("--candidate-root and --output are required for fixture creation.")
    result = prepare(
        args.candidate_root,
        args.output,
        input_start=args.input_start,
        input_count=args.input_count,
        output_start=args.output_start,
        output_count=args.output_count,
        payload_mode=args.payload_mode,
    )
    print(
        f"Prepared {args.payload_mode} wrapper fixtures: "
        + ", ".join(
            f"{scheme}={row['retained_values']}" for scheme, row in result["files"].items()
        )
    )


if __name__ == "__main__":
    main()
