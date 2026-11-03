#!/usr/bin/env python3
"""Create and verify immutable OpenEye wrapper evidence bundles."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile


SCHEMA_VERSION = 1
METHOD = "openeye-wrapper-evidence-bundle"
SCHEMES = ("4:4", "3:4", "2:4")
RUN_FILES = (
    "SHA256SUMS",
    "hardware_metrics.json",
    "regression_manifest.txt",
    "wrapper-conv-candidates-layer.log",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def parse_key_values(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Invalid key=value line {line_number} in {path}.")
        key, value = line.split("=", 1)
        if not key or key in result:
            raise ValueError(f"Missing or duplicate key {key!r} in {path}.")
        result[key] = value
    return result


def safe_relative(value: str, label: str) -> Path:
    pure = PurePosixPath(value)
    if not value or pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"Unsafe relative path for {label}: {value!r}")
    return Path(*pure.parts)


def file_record(root: Path, relative: Path) -> dict[str, object]:
    path = root / relative
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular file: {path}")
    return {"bytes": path.stat().st_size, "sha256": sha256(path)}


def parse_sha256s(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = raw.split(maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise ValueError(f"Invalid SHA256SUMS line {line_number} in {path}.")
        relative = parts[1].lstrip(" *")
        safe_relative(relative, f"SHA256SUMS line {line_number}")
        if relative in rows:
            raise ValueError(f"Duplicate SHA256SUMS entry {relative!r} in {path}.")
        rows[relative] = parts[0]
    return rows


def copy_regular(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"Refusing to archive a non-regular file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def verify_fixture_manifest(path: Path) -> dict:
    manifest = load_json(path)
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported fixture manifest schema.")
    if manifest.get("method") != "openeye-wrapper-actual-candidate-int8-fixtures":
        raise ValueError("Unexpected fixture method.")
    if manifest.get("source_layer") != "layer1.0.conv1":
        raise ValueError("Unexpected source layer in fixture manifest.")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(SCHEMES):
        raise ValueError("Fixture manifest must contain exactly 4:4, 3:4 and 2:4.")
    for scheme, row in files.items():
        if not isinstance(row, dict):
            raise ValueError(f"Invalid fixture record for {scheme}.")
        safe_relative(str(row.get("file", "")), f"fixture {scheme}")
        if not isinstance(row.get("bytes"), int) or not isinstance(row.get("sha256"), str):
            raise ValueError(f"Incomplete fixture record for {scheme}.")
    for field in ("source_candidate_manifest_sha256", "source_checkpoint_sha256"):
        if not isinstance(manifest.get(field), str) or len(manifest[field]) != 64:
            raise ValueError(f"Fixture manifest has no valid {field}.")
    return manifest


def expected_run_scheme(scheme: str) -> str:
    return scheme.replace(":", "x")


def validate_run(source: Path, scheme: str, fixture_sha: str) -> tuple[dict, dict]:
    if not source.is_dir():
        raise ValueError(f"Run directory does not exist: {source}")
    if any(not (source / name).is_file() for name in RUN_FILES):
        raise ValueError(f"Run directory is incomplete: {source}")
    checksums = parse_sha256s(source / "SHA256SUMS")
    expected_names = set(RUN_FILES) - {"SHA256SUMS"}
    if set(checksums) != expected_names:
        raise ValueError(f"Unexpected SHA256SUMS file set in {source}.")
    for name, expected in checksums.items():
        if sha256(source / name) != expected:
            raise ValueError(f"Source run checksum mismatch: {source / name}")

    regression = parse_key_values(source / "regression_manifest.txt")
    scheme_x = expected_run_scheme(scheme)
    required = {
        "measurement_level": "rtl-wrapper-simulation",
        "mode": "candidates-layer",
        "architecture_id": "openeye-fpga-first-layer-v6-staging-depths",
        "fixture_manifest_sha256": fixture_sha,
        "fixture_payload_mode": "int8-symmetric",
        "candidate_schemes": scheme_x,
        "geometry": "56x56-ch64-f64",
        "result": "PASS",
        "exit_status": "0",
        "metrics_validation": "PASS",
    }
    for key, expected in required.items():
        if regression.get(key) != expected:
            raise ValueError(
                f"Run {scheme} has {key}={regression.get(key)!r}, expected {expected!r}."
            )
    openeye_commit = regression.get("openeye_commit", "")
    if len(openeye_commit) != 40:
        raise ValueError(f"Run {scheme} has no full OpenEye commit.")

    metrics = load_json(source / "hardware_metrics.json")
    if metrics.get("schema_version") != 1:
        raise ValueError(f"Run {scheme} has unsupported metrics schema.")
    if metrics.get("validation", {}).get("result") != "PASS":
        raise ValueError(f"Run {scheme} metrics did not pass validation.")
    by_scheme = metrics.get("by_scheme")
    if not isinstance(by_scheme, dict) or set(by_scheme) != {scheme_x}:
        raise ValueError(f"Run {scheme} metrics contain the wrong scheme.")
    summary = by_scheme[scheme_x]
    for field in (
        "accepted_weight_bytes",
        "accepted_weight_words",
        "compute_cycles",
        "load_cycles",
        "output_accepted_words",
        "stall_cycles",
        "total_cycles",
    ):
        if not isinstance(summary.get(field), int):
            raise ValueError(f"Run {scheme} has no integer metric {field}.")
    return regression, summary


def parse_run_argument(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must use SCHEME=PATH.")
    scheme, path = value.split("=", 1)
    if scheme not in SCHEMES:
        raise argparse.ArgumentTypeError(f"Unsupported scheme {scheme!r}.")
    return scheme, Path(path).expanduser().resolve()


def create_bundle(args: argparse.Namespace) -> dict:
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite bundle: {output}")
    if len(args.domino_commit) != 40 or any(
        character not in "0123456789abcdef" for character in args.domino_commit
    ):
        raise ValueError("--domino-commit must be a full lowercase Git commit.")
    output.parent.mkdir(parents=True, exist_ok=True)
    runs = dict(args.run)
    if set(runs) != set(SCHEMES) or len(args.run) != len(SCHEMES):
        raise ValueError("Exactly one --run is required for each of 4:4, 3:4 and 2:4.")

    fixture_root = args.fixture_root.expanduser().resolve()
    fixture_manifest_path = fixture_root / "manifest.json"
    fixture = verify_fixture_manifest(fixture_manifest_path)
    fixture_manifest_sha = sha256(fixture_manifest_path)
    run_inputs: dict[str, tuple[Path, dict, dict]] = {}
    openeye_commits: set[str] = set()
    for scheme in SCHEMES:
        regression, metrics = validate_run(runs[scheme], scheme, fixture_manifest_sha)
        openeye_commits.add(regression["openeye_commit"])
        run_inputs[scheme] = (runs[scheme], regression, metrics)
    if len(openeye_commits) != 1:
        raise ValueError("The three runs use different OpenEye commits.")

    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        archived_files: list[Path] = []
        fixture_destination = temporary / "fixture"
        copy_regular(fixture_manifest_path, fixture_destination / "manifest.json")
        archived_files.append(Path("fixture/manifest.json"))
        for scheme in SCHEMES:
            row = fixture["files"][scheme]
            relative = safe_relative(row["file"], f"fixture {scheme}")
            source = fixture_root / relative
            if source.stat().st_size != row["bytes"] or sha256(source) != row["sha256"]:
                raise ValueError(f"Fixture payload mismatch for {scheme}: {source}")
            destination_relative = Path("fixture") / relative
            copy_regular(source, temporary / destination_relative)
            archived_files.append(destination_relative)

        manifest_runs: dict[str, dict] = {}
        for scheme in SCHEMES:
            source, regression, metrics = run_inputs[scheme]
            run_relative = Path("runs") / expected_run_scheme(scheme)
            for name in RUN_FILES:
                destination_relative = run_relative / name
                copy_regular(source / name, temporary / destination_relative)
                archived_files.append(destination_relative)
            manifest_runs[scheme] = {
                "directory": run_relative.as_posix(),
                "completed_utc": regression.get("completed_utc"),
                "simulator": regression.get("simulator"),
                "sim_timeout_ns": int(regression["sim_timeout_ns"]),
                "timeout_seconds": int(regression["timeout_seconds"]),
                "metrics": metrics,
                "reproduction_command": (
                    f"OPENEYE_SIMULATOR={regression['simulator']} "
                    f"OPENEYE_CANDIDATE_SCHEMES='{expected_run_scheme(scheme)}' "
                    f"OPENEYE_SIM_TIMEOUT_NS={regression['sim_timeout_ns']} "
                    f"OPENEYE_TEST_TIMEOUT_SECONDS={regression['timeout_seconds']} "
                    "fpga/openeye/run_wrapper_conv_regression.sh "
                    "~/Projects/OpenEye OUTPUT_DIR candidates-layer FIXTURE_ROOT"
                ),
            }

        files = {
            relative.as_posix(): file_record(temporary, relative)
            for relative in sorted(archived_files, key=lambda value: value.as_posix())
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "method": METHOD,
            "bundle_id": args.bundle_id,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "claim_level": "rtl-wrapper-simulation-correctness-and-counters",
            "source": {
                "domino_branch": args.domino_branch,
                "domino_commit": args.domino_commit,
                "openeye_commit": openeye_commits.pop(),
            },
            "architecture_id": "openeye-fpga-first-layer-v6-staging-depths",
            "source_layer": fixture["source_layer"],
            "fixture": {
                "manifest": "fixture/manifest.json",
                "manifest_sha256": fixture_manifest_sha,
                "payload_mode": fixture["payload_mode"],
                "quantization": fixture["quantization"],
                "source_candidate_manifest_sha256": fixture[
                    "source_candidate_manifest_sha256"
                ],
                "source_checkpoint_sha256": fixture["source_checkpoint_sha256"],
            },
            "runs": manifest_runs,
            "files": files,
            "limitations": [
                "This bundle proves RTL wrapper correctness and records RTL counters.",
                "It does not prove full-model INT8 accuracy, board latency, power, or speedup.",
            ],
        }
        manifest_path = temporary / "bundle_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        manifest_hash = sha256(manifest_path)
        (temporary / "BUNDLE_MANIFEST.sha256").write_text(
            f"{manifest_hash}  bundle_manifest.json\n", encoding="utf-8"
        )
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary)
        raise
    return verify_bundle(output)


def verify_record(root: Path, relative_text: str, row: dict) -> None:
    relative = safe_relative(relative_text, "bundle file")
    path = root / relative
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Missing regular bundle file: {relative_text}")
    if path.stat().st_size != row.get("bytes"):
        raise ValueError(f"Size mismatch for {relative_text}.")
    if sha256(path) != row.get("sha256"):
        raise ValueError(f"SHA-256 mismatch for {relative_text}.")


def verify_bundle(bundle_root: Path) -> dict:
    root = bundle_root.expanduser().resolve()
    manifest_path = root / "bundle_manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("method") != METHOD:
        raise ValueError("Unsupported OpenEye evidence bundle manifest.")
    checksum_lines = parse_sha256s(root / "BUNDLE_MANIFEST.sha256")
    expected_manifest_hash = checksum_lines.get("bundle_manifest.json")
    if set(checksum_lines) != {"bundle_manifest.json"} or sha256(manifest_path) != expected_manifest_hash:
        raise ValueError("Bundle manifest root checksum mismatch.")

    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Bundle manifest has no files.")
    for relative, row in files.items():
        if not isinstance(row, dict):
            raise ValueError(f"Invalid file record for {relative}.")
        verify_record(root, relative, row)
    entries = list(root.rglob("*"))
    symlinks = [path.relative_to(root).as_posix() for path in entries if path.is_symlink()]
    if symlinks:
        raise ValueError(f"Bundle contains symlinks: {sorted(symlinks)}.")
    actual_files = {
        path.relative_to(root).as_posix()
        for path in entries
        if path.is_file()
    }
    expected_files = set(files) | {"bundle_manifest.json", "BUNDLE_MANIFEST.sha256"}
    if actual_files != expected_files:
        raise ValueError(
            f"Bundle file set mismatch; extra={sorted(actual_files - expected_files)}, "
            f"missing={sorted(expected_files - actual_files)}."
        )

    fixture_relative = safe_relative(manifest["fixture"]["manifest"], "fixture manifest")
    fixture_path = root / fixture_relative
    fixture = verify_fixture_manifest(fixture_path)
    if sha256(fixture_path) != manifest["fixture"]["manifest_sha256"]:
        raise ValueError("Fixture manifest identity mismatch.")
    for field in ("source_candidate_manifest_sha256", "source_checkpoint_sha256"):
        if fixture[field] != manifest["fixture"][field]:
            raise ValueError(f"Fixture {field} differs from bundle identity.")
    for scheme, row in fixture["files"].items():
        path = root / "fixture" / safe_relative(row["file"], f"fixture {scheme}")
        if path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
            raise ValueError(f"Archived fixture payload mismatch for {scheme}.")

    runs = manifest.get("runs")
    if not isinstance(runs, dict) or set(runs) != set(SCHEMES):
        raise ValueError("Bundle must contain exactly the three required schemes.")
    summaries: dict[str, dict] = {}
    openeye_commit = manifest.get("source", {}).get("openeye_commit")
    for scheme in SCHEMES:
        run_root = root / safe_relative(runs[scheme]["directory"], f"run {scheme}")
        regression, metrics = validate_run(
            run_root, scheme, manifest["fixture"]["manifest_sha256"]
        )
        if regression["openeye_commit"] != openeye_commit:
            raise ValueError(f"Run {scheme} OpenEye commit differs from bundle identity.")
        if metrics != runs[scheme]["metrics"]:
            raise ValueError(f"Run {scheme} metrics differ from bundle summary.")
        summaries[scheme] = metrics

    return {
        "schema_version": 1,
        "method": "openeye-wrapper-evidence-integrity",
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "bundle_id": manifest["bundle_id"],
        "bundle_manifest_sha256": sha256(manifest_path),
        "source": manifest["source"],
        "source_layer": manifest["source_layer"],
        "source_candidate_manifest_sha256": manifest["fixture"][
            "source_candidate_manifest_sha256"
        ],
        "source_checkpoint_sha256": manifest["fixture"]["source_checkpoint_sha256"],
        "schemes": summaries,
        "result": "PASS",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create", help="Create a new immutable bundle.")
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--bundle-id", required=True)
    create.add_argument("--fixture-root", type=Path, required=True)
    create.add_argument("--run", action="append", type=parse_run_argument, required=True)
    create.add_argument("--domino-branch", required=True)
    create.add_argument("--domino-commit", required=True)

    verify = subparsers.add_parser("verify", help="Verify a copied bundle.")
    verify.add_argument("--bundle-root", type=Path, required=True)
    verify.add_argument("--report", type=Path)
    return parser.parse_args()


def render_result(result: dict, report: Path | None = None) -> None:
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if report is None:
        print(rendered, end="")
        return
    destination = report.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite report: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(rendered, encoding="utf-8")
    print(f"OpenEye evidence verification passed: {destination}")


def main() -> None:
    args = parse_args()
    if args.command == "create":
        render_result(create_bundle(args))
    else:
        render_result(verify_bundle(args.bundle_root), args.report)


if __name__ == "__main__":
    main()
