#!/usr/bin/env python3
"""Publish allowlisted metadata and numeric counters, never simulator payloads.

Operate on a fresh copy AFTER private checkpoint sealing. Full raw simulator
logs remain runner-private; published logs reconstruct only the records consumed
by the unchanged counter verifier. Failing JUnit keeps failure status but drops
free-text details. This transformation cannot promote a failed gate to PASS.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fpga.openeye.network_checkpoint import public_path
from fpga.openeye.summarize_wrapper_metrics import parse_log

PRIVATE_KEYS = {
    "weights",
    "weight_values",
    "activations",
    "actual",
    "reference",
    "original_actual",
    "original_reference",
    "coefficients",
    "multiplier",
    "bias",
    "fmap",
    "token",
    "refresh_token",
    "access_token",
}


def validate_metadata(value):
    if isinstance(value, dict):
        if PRIVATE_KEYS.intersection(value):
            raise ValueError("Private tensor/coefficient/credential key in metadata")
        for child in value.values():
            validate_metadata(child)
    elif isinstance(value, list):
        for child in value:
            validate_metadata(child)


def numeric_record(value):
    if isinstance(value, dict):
        if any(not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", key) for key in value):
            raise ValueError("Unexpected metrics key")
        for child in value.values():
            numeric_record(child)
    elif (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError("Metrics may contain numeric counters only")


def counter_log(source):
    records, counts = parse_log(source)
    lines = []
    for count in counts:
        lines.append(
            "Host will send: iact {iact}, wght {wght}, psum {psum}, quantize {quantize} words".format(
                **count
            )
        )
    for record in records:
        numeric_record(record)
        lines.append(
            "DOMINO_HW_METRICS " + json.dumps(record, sort_keys=True, allow_nan=False)
        )
    return "\n".join(lines) + "\n"


def safe_junit(source):
    root = ET.fromstring(source.read_bytes())
    for node in root.iter():
        node.text = None
        node.tail = None
        allowed = {
            "testsuites": {"name"},
            "testsuite": {"name", "tests", "failures", "errors", "skipped", "time"},
            "testcase": {"classname", "name", "time"},
            "properties": set(),
            "property": {"name", "value"},
            "failure": set(),
            "error": set(),
            "skipped": set(),
            "system-out": set(),
            "system-err": set(),
        }.get(node.tag)
        if allowed is None:
            raise ValueError("Unknown JUnit element")
        node.attrib = {k: v for k, v in node.attrib.items() if k in allowed}
        if node.tag == "property":
            name = node.get("name", "")
            value = node.get("value", "")
            if name not in {
                "cocotb",
                "random_seed",
                "sim_time_unit",
                "sim_time_start",
                "sim_time_stop",
                "sim_time_duration",
                "sim_time_ratio",
            }:
                node.attrib = {"name": "private-property-omitted"}
            elif not re.fullmatch(r"(?:True|False|ns|us|ps|s|[0-9.eE+\-]+)", value):
                raise ValueError("Unexpected JUnit property")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def publish(source, destination):
    if source.is_symlink() or destination.exists() or source == destination:
        raise ValueError("Require regular evidence and a new public destination")
    if source in destination.parents or destination in source.parents:
        raise ValueError("Public copy must be disjoint from private evidence")
    destination.mkdir(parents=True)
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ValueError("Symlink in evidence")
        if not path.is_file() or path.name == "SHA256SUMS":
            continue
        relative = str(path.relative_to(source))
        public_path(relative)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".log":
            target.write_text(counter_log(path))
        elif path.suffix == ".xml":
            target.write_bytes(safe_junit(path))
        elif path.suffix == ".json":
            value = json.loads(path.read_text())
            validate_metadata(value)
            target.write_bytes(path.read_bytes())
        elif path.name == "regression_manifest.txt":
            target.write_bytes(path.read_bytes())
        else:
            raise ValueError("Unsupported public metadata type")
    # Each nested layer checksum is regenerated after redaction, then root.
    roots = sorted(
        {p.parent for p in destination.rglob("regression_manifest.txt")},
        key=lambda p: len(p.parts),
        reverse=True,
    )
    for root in [*roots, destination]:
        files = sorted(
            p for p in root.rglob("*") if p.is_file() and p.name != "SHA256SUMS"
        )
        (root / "SHA256SUMS").write_text(
            "".join(
                hashlib.sha256(p.read_bytes()).hexdigest()
                + "  "
                + str(p.relative_to(root))
                + "\n"
                for p in files
            )
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output_existed = args.output.exists()
    try:
        publish(args.source.resolve(), args.output.resolve())
    except (ValueError, OSError, ET.ParseError):
        # Never echo payload-bearing parser or validation details to public logs.
        if not output_existed:
            shutil.rmtree(args.output, ignore_errors=True)
        print(
            "Public metadata export refused; no private details published",
            file=sys.stderr,
        )
        raise SystemExit(2)
    print("Public metadata/counter copy created; simulator payload logs withheld")


if __name__ == "__main__":
    main()
