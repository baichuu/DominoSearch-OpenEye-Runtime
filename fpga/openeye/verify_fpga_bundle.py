#!/usr/bin/env python3
"""Verify a downloaded Domino packed-v1 FPGA artifact bundle.

This verifier intentionally uses only the Python standard library.  It checks
transport integrity and N:M structural invariants before an artifact is handed
to an RTL mapper; it does not repeat the PyTorch numerical/model verification
recorded in ``packed-verification.json``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath


FORMAT = {"name": "domino-packed-nm-bitmask", "version": 1}
DTYPE_BYTES = {
    "bool": 1,
    "int8": 1,
    "uint8": 1,
    "int16": 2,
    "uint16": 2,
    "float16": 2,
    "bfloat16": 2,
    "int32": 4,
    "uint32": 4,
    "float32": 4,
    "int64": 8,
    "uint64": 8,
    "float64": 8,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return data


def checked_file(root: Path, row: dict, label: str) -> Path:
    relative = row.get("file")
    if not isinstance(relative, str):
        raise ValueError(f"{label} has no file name.")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"Unsafe relative path for {label}: {relative}")
    path = root.joinpath(*pure.parts)
    if not path.is_file():
        raise ValueError(f"Missing {label}: {path}")
    expected_bytes = int(row["bytes"])
    if path.stat().st_size != expected_bytes:
        raise ValueError(
            f"Size mismatch for {label}: {path.stat().st_size} != {expected_bytes}."
        )
    actual_hash = sha256(path)
    if actual_hash != row["sha256"]:
        raise ValueError(f"SHA-256 mismatch for {label}: {path}")
    return path


def expected_tensor_bytes(dtype: str, elements: int, label: str) -> int:
    try:
        item_size = DTYPE_BYTES[dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype for {label}: {dtype}") from exc
    return item_size * elements


def verify_metadata(path: Path, n: int, m: int, groups: int) -> None:
    if m != 4:
        raise ValueError(f"FPGA milestone accepts only M=4, got {n}:{m}.")
    count = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            for mask in chunk:
                count += 1
                if mask >> m:
                    raise ValueError(
                        f"Metadata group {count - 1} has bits outside M={m}: 0x{mask:02x}."
                    )
                if mask.bit_count() != n:
                    raise ValueError(
                        f"Metadata group {count - 1} retains {mask.bit_count()}, expected {n}."
                    )
    if count != groups:
        raise ValueError(f"Metadata contains {count} groups, expected {groups}.")


def verify_packed_manifest(root: Path, expected: dict) -> dict:
    manifest_path = root / "manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != 1 or manifest.get("format") != FORMAT:
        raise ValueError(f"Unsupported packed manifest: {manifest_path}")
    if sha256(manifest_path) != expected["packed_manifest_sha256"]:
        raise ValueError(f"Packed manifest SHA-256 mismatch: {manifest_path}")
    if manifest.get("checkpoint", {}).get("sha256") != expected["checkpoint_sha256"]:
        raise ValueError("Checkpoint identity differs between bundle and packed manifest.")
    if manifest.get("scheme", {}).get("sha256") != expected["scheme_sha256"]:
        raise ValueError("Scheme identity differs between bundle and packed manifest.")

    layers = manifest.get("layers")
    scheme = manifest.get("scheme", {}).get("content")
    if not isinstance(layers, list) or len(layers) != int(expected["layers"]):
        raise ValueError("Packed layer count differs from the bundle manifest.")
    if not isinstance(scheme, dict):
        raise ValueError("Packed manifest has no embedded scheme.")
    layer_names = {row.get("name") for row in layers}
    if len(layer_names) != len(layers) or layer_names != set(scheme):
        raise ValueError("Layer names do not exactly match the embedded scheme.")

    counts: dict[str, int] = {}
    for index, row in enumerate(layers):
        label = f"layer {index} ({row.get('name')})"
        if row.get("index") != index:
            raise ValueError(f"Non-contiguous layer index in {label}.")
        n, m = int(row["n"]), int(row["m"])
        if m != 4 or n not in {2, 3, 4}:
            raise ValueError(f"Unsupported N:M in {label}: {n}:{m}.")
        if scheme[row["name"]] != [n, m]:
            raise ValueError(f"Embedded scheme differs from row for {label}.")
        shape = row["source_weight_shape"]
        elements = math.prod(int(value) for value in shape)
        groups = int(row["groups"])
        if groups * m != elements:
            raise ValueError(f"Shape/group mismatch in {label}.")
        value_count = int(row["value_count"])
        if value_count != groups * n:
            raise ValueError(f"Packed value count mismatch in {label}.")
        expected_density = n / m
        if not math.isclose(float(row["density"]), expected_density, abs_tol=1e-12):
            raise ValueError(f"Density mismatch in {label}.")

        values = row["values"]
        if int(values["bytes"]) != expected_tensor_bytes(
            values["dtype"], value_count, label
        ):
            raise ValueError(f"Value byte count is inconsistent in {label}.")
        checked_file(root, values, f"{label} values")

        metadata = row.get("metadata")
        if n == m:
            if metadata is not None:
                raise ValueError(f"Dense candidate unexpectedly has metadata in {label}.")
        else:
            if not isinstance(metadata, dict):
                raise ValueError(f"Sparse candidate has no metadata in {label}.")
            if (
                metadata.get("dtype") != "uint8"
                or metadata.get("encoding") != "one-position-bitmask-per-group"
                or int(metadata.get("storage_bits_per_group", 0)) != 8
                or int(metadata["bytes"]) != groups
            ):
                raise ValueError(f"Unsupported metadata encoding in {label}.")
            metadata_path = checked_file(root, metadata, f"{label} metadata")
            verify_metadata(metadata_path, n, m, groups)
        counts[f"{n}:{m}"] = counts.get(f"{n}:{m}", 0) + 1

    auxiliary = manifest.get("auxiliary_tensors")
    if not isinstance(auxiliary, list):
        raise ValueError("Packed manifest has no auxiliary tensor list.")
    state_keys = {row.get("state_key") for row in auxiliary}
    if len(state_keys) != len(auxiliary):
        raise ValueError("Auxiliary tensor state keys are not unique.")
    for index, row in enumerate(auxiliary):
        label = f"auxiliary tensor {index} ({row.get('state_key')})"
        elements = int(row["elements"])
        if math.prod(int(value) for value in row["shape"]) != elements:
            raise ValueError(f"Shape/element mismatch for {label}.")
        if int(row["bytes"]) != expected_tensor_bytes(row["dtype"], elements, label):
            raise ValueError(f"Byte count mismatch for {label}.")
        checked_file(root, row, label)

    return {
        "layers": len(layers),
        "auxiliary_tensors": len(auxiliary),
        "schemes": counts,
        "checkpoint_sha256": expected["checkpoint_sha256"],
        "packed_manifest_sha256": expected["packed_manifest_sha256"],
    }


def verify_candidate(bundle_root: Path, name: str, expected: dict) -> dict:
    candidate_root = bundle_root / name
    report_path = candidate_root / "packed-verification.json"
    if sha256(report_path) != expected["verification_report_sha256"]:
        raise ValueError(f"Verification report SHA-256 mismatch for {name}.")
    report = load_json(report_path)
    for field in ("checkpoint_sha256", "packed_manifest_sha256", "scheme_sha256"):
        expected_field = expected[field]
        if report.get(field) != expected_field:
            raise ValueError(f"{field} differs in the {name} verification report.")
    if report.get("full_model") != expected["full_model"]:
        raise ValueError(f"Full-model verification result differs for {name}.")
    if report.get("samples") != expected["samples"]:
        raise ValueError(f"Verification sample count differs for {name}.")
    if not all(row.get("passed") is True for row in report.get("layers", [])):
        raise ValueError(f"A recorded layer verification failed for {name}.")

    vectors = report.get("test_vectors")
    if not isinstance(vectors, dict):
        raise ValueError(f"No test-vector record for {name}.")
    vector_root = candidate_root / f"test-vectors-{expected['samples']}"
    for kind in ("input", "output"):
        checked_file(
            vector_root,
            {
                **vectors[kind],
                "bytes": math.prod(vectors[f"{kind}_shape"])
                * DTYPE_BYTES[vectors["dtype"]],
            },
            f"{name} {kind} test vector",
        )

    summary = verify_packed_manifest(candidate_root / "packed-v1", expected)
    summary["recorded_full_model_check"] = expected["full_model"]
    summary["recorded_samples"] = expected["samples"]
    return summary


def verify_bundle(bundle_root: Path) -> dict:
    bundle_root = bundle_root.expanduser().resolve()
    manifest_path = bundle_root / "bundle-manifest.json"
    bundle = load_json(manifest_path)
    if bundle.get("schema_version") != 1:
        raise ValueError("Unsupported bundle manifest schema.")
    for archive_name, row in bundle.get("archives", {}).items():
        checked_file(bundle_root, {**row, "file": archive_name}, f"archive {archive_name}")
    candidates = bundle.get("run", {}).get("candidates")
    if not isinstance(candidates, dict) or not candidates:
        raise ValueError("Bundle has no candidates.")
    return {
        "schema_version": 1,
        "method": "packed-v1-fpga-ingress-integrity",
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "bundle_manifest_sha256": sha256(manifest_path),
        "source_branch": bundle["run"].get("branch"),
        "source_commit": bundle["run"].get("commit"),
        "candidates": {
            name: verify_candidate(bundle_root, name, expected)
            for name, expected in sorted(candidates.items())
        },
        "claim_level": "transport-and-structure-only",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = verify_bundle(args.bundle_root)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.report:
        report = args.report.expanduser().resolve()
        if report.exists():
            raise FileExistsError(f"Refusing to overwrite report: {report}")
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(rendered, encoding="utf-8")
        print(f"Bundle verification passed: {report}")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
