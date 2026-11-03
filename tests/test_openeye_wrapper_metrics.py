import json
from pathlib import Path
import subprocess
import sys

from fpga.openeye.summarize_wrapper_metrics import build_summary


def test_cli_does_not_admit_failed_rtl_into_cost_table(tmp_path):
    log = tmp_path / "wrapper.log"
    output = tmp_path / "metrics.json"
    log.write_text("Host will send: iact 2, wght 12, psum 4, quantize 8 words\n"
                   f"DOMINO_HW_METRICS {json.dumps(_record(12))}\n")
    subprocess.run([sys.executable, "fpga/openeye/summarize_wrapper_metrics.py",
                    "--log", str(log), "--output", str(output), "--mode", "kernel7-smoke",
                    "--architecture-id", "openeye-fpga-full-layer-r3-staging-depths",
                    "--rtl-result", "FAIL"], check=True)
    summary = json.loads(output.read_text())
    assert summary["validation"]["result"] == "PASS"  # Counters only.
    assert summary["rtl_correctness"] == "FAIL"
    assert not summary["eligible_for_cost_table"]
    assert summary["architecture_id"] == "openeye-fpga-full-layer-r3-staging-depths"


def _record(weight_words: int) -> dict:
    return {
        "schema_version": 1,
        "dma_bitwidth": 64,
        "total_cycles": 15,
        "load_cycles": 10,
        "compute_cycles": 5,
        "input_accepted_words": weight_words + 3,
        "input_stall_cycles": 2,
        "output_accepted_words": 4,
        "output_stall_cycles": 1,
        "stall_cycles": 3,
        "accepted_weight_words": weight_words,
        "accepted_weight_bytes": weight_words * 8,
        "state_cycles": {"GET_PARAMETERS": 10, "WAIT_FOR_RESULTS": 5},
        "input_accepted_words_by_state": {
            "GET_PARAMETERS": 3,
            "GET_WGHT": weight_words,
        },
    }


def test_synthetic_kernel_smoke_is_not_a_layer_cost(tmp_path):
    log = tmp_path / "wrapper.log"
    output = tmp_path / "metrics.json"
    log.write_text("Host will send: iact 2, wght 12, psum 4, quantize 8 words\n"
                   f"DOMINO_HW_METRICS {json.dumps(_record(12))}\n")
    subprocess.run([sys.executable, "fpga/openeye/summarize_wrapper_metrics.py",
                    "--log", str(log), "--output", str(output), "--mode", "kernel7-smoke",
                    "--rtl-result", "PASS"], check=True)
    summary = json.loads(output.read_text())
    assert summary["rtl_correctness"] == "PASS"
    assert not summary["eligible_for_cost_table"]


def test_tiled_layer_aggregates_all_reload_costs(tmp_path):
    log = tmp_path / "wrapper.log"
    log.write_text("".join("Host will send: iact 2, wght 12, psum 4, quantize 8 words\n"
                          f"DOMINO_HW_METRICS {json.dumps(_record(12))}\n" for _ in range(4)))
    summary, valid = build_summary(log, "packed-layer", ["4:4"] * 4)
    assert valid
    assert summary["by_scheme"]["4:4"]["executions"] == 4
    assert summary["by_scheme"]["4:4"]["total_cycles"] == 60
    assert summary["by_scheme"]["4:4"]["accepted_weight_bytes"] == 384


def test_runtime_pilot_retains_valid_counters_but_is_not_cost_table(tmp_path):
    log = tmp_path / "wrapper.log"
    output = tmp_path / "metrics.json"
    log.write_text("Host will send: iact 2, wght 12, psum 4, quantize 8 words\n"
                   f"DOMINO_HW_METRICS {json.dumps(_record(12))}\n")
    subprocess.run([sys.executable, "fpga/openeye/summarize_wrapper_metrics.py",
                    "--log", str(log), "--output", str(output), "--mode", "packed-layer",
                    "--rtl-result", "PASS", "--correctness-only"], check=True)
    summary = json.loads(output.read_text())
    assert summary["validation"]["result"] == "PASS"
    assert summary["rtl_correctness"] == "PASS"
    assert not summary["eligible_for_cost_table"]
    assert summary["records"][0]["total_cycles"] == 15
    assert "runtime correctness pilot" in summary["cost_exclusion_reason"]


def test_build_summary_labels_and_validates_handshake_counts(tmp_path: Path) -> None:
    record = _record(12)
    log = tmp_path / "wrapper.log"
    log.write_text(
        "Host will send: iact 2, wght 12, psum 4, quantize 8 words\n"
        f"prefix DOMINO_HW_METRICS {json.dumps(record)}\n",
        encoding="utf-8",
    )

    summary, valid = build_summary(log, "candidates", ["3x4"])

    assert valid
    assert summary["validation"]["result"] == "PASS"
    assert summary["records"][0]["scheme"] == "3x4"
    assert summary["by_scheme"]["3x4"]["accepted_weight_bytes"] == 96


def test_build_summary_rejects_attempted_vs_accepted_weight_mismatch(
    tmp_path: Path,
) -> None:
    log = tmp_path / "wrapper.log"
    log.write_text(
        "Host will send: iact 2, wght 13, psum 4, quantize 8 words\n"
        f"DOMINO_HW_METRICS {json.dumps(_record(12))}\n",
        encoding="utf-8",
    )

    summary, valid = build_summary(log, "candidates", ["3x4"])

    assert not valid
    assert summary["validation"]["result"] == "FAIL"
    assert "accepted weight words 12 != host section 13" in summary["validation"]["errors"][0]


def test_build_summary_accepts_trailing_simulator_text(tmp_path: Path) -> None:
    log = tmp_path / "wrapper.log"
    log.write_text(
        "Host will send: iact 2, wght 12, psum 4, quantize 8 words\n"
        f"DOMINO_HW_METRICS {json.dumps(_record(12))}FST info: dumpfile opened\n",
        encoding="utf-8",
    )

    summary, valid = build_summary(log, "focused", [])

    assert valid
    assert len(summary["records"]) == 1
