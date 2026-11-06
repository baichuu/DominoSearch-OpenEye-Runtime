"""The public runner must never fetch/execute the private source repository."""

import ast
import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PIN = "36395ddc530e106f06c50025a4069dd097688d8d"


def test_python_helpers_imported_by_generated_patches_are_exported():
    """Generated Python imports are dependencies even without a source AST edge."""
    required = set()
    for patch in (ROOT/'fpga/openeye/patches').glob('*.patch'):
        required.update(re.findall(r'from fpga\.openeye\.(\w+) import',patch.read_text()))
    assert 'nm_offset_adapter' in required
    for module in required:
        path=ROOT/'fpga/openeye'/(module+'.py')
        assert path.is_file(),f'Missing generated-runtime helper: {module}'
        compile(path.read_bytes(),str(path),'exec')


def test_manual_standard_runners_and_public_source_only():
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/openeye-full-layer.yml").read_text()
    )
    trigger = workflow.get("on", workflow.get(True))
    assert set(trigger) == {"workflow_dispatch"}
    assert trigger["workflow_dispatch"]["inputs"]["scope"]["options"] == [
        "hybrid-network-resumable"
    ]
    assert workflow["env"]["OPENEYE_PIN"] == PIN
    assert workflow["permissions"] == {"contents": "read"}
    for job in workflow["jobs"].values():
        assert job["runs-on"] == "ubuntu-24.04"
        assert (
            "github.repository == 'baichuu/DominoSearch-OpenEye-Runtime'" in job["if"]
        )
        for step in job["steps"]:
            if step.get("uses", "").startswith("actions/checkout"):
                settings = step["with"]
                assert settings.get(
                    "repository", "baichuu/DominoSearch-OpenEye-Runtime"
                ) in [
                    "baichuu/DominoSearch-OpenEye-Runtime",
                    "Learning-Chips-Lab/OpenEye",
                ]
                assert settings["ref"] in [
                    "${{ github.sha }}",
                    "${{ env.OPENEYE_PIN }}",
                ]
                assert settings.get("persist-credentials") is False


def test_runtime_pin_and_sources_are_separate_from_private_history():
    manifest = json.loads((ROOT / "PUBLIC_SOURCE_MANIFEST.json").read_text())
    assert manifest["openeye_pin"] == PIN
    assert not (ROOT / "assets").exists() and not (ROOT / "models").exists()
    for filename in ["run_network_inference.py", "run_residual_block.py"]:
        tree = ast.parse((ROOT / "fpga/openeye" / filename).read_text())
        pin = next(
            ast.literal_eval(n.value)
            for n in tree.body
            if isinstance(n, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "EXPECTED_OPENEYE_COMMIT"
                for t in n.targets
            )
        )
        assert pin == PIN


def test_private_simulator_logs_and_checkpoint_paths_never_uploaded():
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/openeye-full-layer.yml").read_text()
    )
    for job in workflow["jobs"].values():
        for step in job["steps"]:
            if step.get("uses", "").startswith("actions/upload-artifact"):
                assert step["with"]["path"] in [
                    "${{ runner.temp }}/network-preflight.json",
                    "${{ runner.temp }}/runtime-report",
                    "${{ runner.temp }}/residual-upload",
                    "${{ runner.temp }}/network-report",
                    "${{ runner.temp }}/network-public-evidence",
                ]
            if step["name"] in [
                "Focused network channel, projection, stride and Dense gates",
                "Execute selected hybrid correctness runtime",
                "Execute or resume checked actual DUT block boundary",
            ]:
                assert (
                    '> "$RUNNER_TEMP/private-' in step["run"] and "2>&1" in step["run"]
                )
                assert 'exit "$status"' in step["run"]
