#!/usr/bin/env python3
"""Extract handshake-based OpenEye wrapper measurements from a cocotb log."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


METRICS_MARKER = "DOMINO_HW_METRICS "
HOST_COUNTS_RE = re.compile(
    r"Host will send: iact (?P<iact>\d+), wght (?P<wght>\d+), "
    r"psum (?P<psum>\d+), quantize (?P<quantize>\d+) words"
)


def parse_log(log_path: Path) -> tuple[list[dict], list[dict[str, int]]]:
    records: list[dict] = []
    host_counts: list[dict[str, int]] = []
    for line_number, line in enumerate(
        log_path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
    ):
        if METRICS_MARKER in line:
            payload = line.split(METRICS_MARKER, 1)[1].strip()
            try:
                # Cocotb and the simulator write through different streams.
                # Their flushes can concatenate a trailing simulator message
                # onto the metrics line, so decode one complete JSON value and
                # deliberately ignore text after it.
                record, _ = json.JSONDecoder().raw_decode(payload)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid metrics JSON at {log_path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"metrics JSON at {log_path}:{line_number} is not an object"
                )
            records.append(record)
        match = HOST_COUNTS_RE.search(line)
        if match:
            host_counts.append({key: int(value) for key, value in match.groupdict().items()})
    return records, host_counts


def validate_records(
    records: list[dict], host_counts: list[dict[str, int]], schemes: list[str]
) -> list[str]:
    errors: list[str] = []
    if not records:
        return ["no DOMINO_HW_METRICS records found"]
    if len(host_counts) != len(records):
        errors.append(
            f"host-count record count {len(host_counts)} != metrics record count {len(records)}"
        )
    if schemes and len(schemes) != len(records):
        errors.append(
            f"scheme count {len(schemes)} != metrics record count {len(records)}"
        )

    required = (
        "dma_bitwidth",
        "total_cycles",
        "load_cycles",
        "compute_cycles",
        "input_accepted_words",
        "accepted_weight_words",
        "accepted_weight_bytes",
        "output_accepted_words",
        "input_stall_cycles",
        "output_stall_cycles",
        "stall_cycles",
        "state_cycles",
        "input_accepted_words_by_state",
    )
    for index, record in enumerate(records):
        missing = [key for key in required if key not in record]
        if missing:
            errors.append(f"record {index} missing fields: {', '.join(missing)}")
            continue
        state_total = sum(record["state_cycles"].values())
        if state_total != record["total_cycles"]:
            errors.append(
                f"record {index} state-cycle sum {state_total} != total_cycles "
                f"{record['total_cycles']}"
            )
        if record["load_cycles"] + record["compute_cycles"] != record["total_cycles"]:
            errors.append(
                f"record {index} load+compute does not equal total_cycles"
            )
        accepted_by_state = sum(record["input_accepted_words_by_state"].values())
        if accepted_by_state != record["input_accepted_words"]:
            errors.append(
                f"record {index} accepted input by-state sum {accepted_by_state} != "
                f"input_accepted_words {record['input_accepted_words']}"
            )
        bytes_per_word = record["dma_bitwidth"] // 8
        expected_weight_bytes = record["accepted_weight_words"] * bytes_per_word
        if expected_weight_bytes != record["accepted_weight_bytes"]:
            errors.append(
                f"record {index} accepted_weight_bytes {record['accepted_weight_bytes']} != "
                f"words*bytes {expected_weight_bytes}"
            )
        if record["stall_cycles"] != (
            record["input_stall_cycles"] + record["output_stall_cycles"]
        ):
            errors.append(f"record {index} stall_cycles is inconsistent")
        if index < len(host_counts):
            expected_words = host_counts[index]["wght"]
            if record["accepted_weight_words"] != expected_words:
                errors.append(
                    f"record {index} accepted weight words "
                    f"{record['accepted_weight_words']} != host section {expected_words}"
                )
    return errors


def build_summary(
    log_path: Path, mode: str, schemes: list[str]
) -> tuple[dict, bool]:
    records, host_counts = parse_log(log_path)
    errors = validate_records(records, host_counts, schemes)

    labelled_records = []
    aggregate: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "executions": 0,
            "accepted_weight_words": 0,
            "accepted_weight_bytes": 0,
            "load_cycles": 0,
            "compute_cycles": 0,
            "stall_cycles": 0,
            "total_cycles": 0,
            "output_accepted_words": 0,
        }
    )
    for index, record in enumerate(records):
        labelled = dict(record)
        label = schemes[index] if index < len(schemes) else f"execution-{index}"
        labelled["scheme"] = label
        labelled_records.append(labelled)
        target = aggregate[label]
        target["executions"] += 1
        for key in target:
            if key != "executions":
                target[key] += int(record.get(key, 0))

    summary = {
        "schema_version": 1,
        "measurement_level": "rtl-wrapper-simulation",
        "mode": mode,
        "source_log": str(log_path.resolve()),
        "validation": {"result": "PASS" if not errors else "FAIL", "errors": errors},
        "records": labelled_records,
        "by_scheme": dict(aggregate),
        "host_section_counts": host_counts,
    }
    return summary, not errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--schemes", nargs="*", default=[])
    parser.add_argument("--architecture-id", help="Explicit hardware identity, not a timing/board claim.")
    parser.add_argument("--rtl-result", choices=("PASS", "FAIL"),
                        help="Simulator correctness result, separate from counter validation.")
    parser.add_argument("--correctness-only", action="store_true",
                        help="Exclude debug runtime/crop pilots from hardware cost tables.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary, valid = build_summary(args.log, args.mode, args.schemes)
    if args.architecture_id:
        summary["architecture_id"] = args.architecture_id
    if args.rtl_result:
        summary["rtl_correctness"] = args.rtl_result
        summary["eligible_for_cost_table"] = (
            valid and args.rtl_result == "PASS" and args.mode != "kernel7-smoke"
            and not args.correctness_only
        )
    if args.correctness_only:
        summary["eligible_for_cost_table"] = False
        summary["cost_exclusion_reason"] = "runtime correctness pilot; not network latency or cost"
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
