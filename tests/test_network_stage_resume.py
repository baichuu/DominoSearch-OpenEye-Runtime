"""Fabricated stage-boundary orchestration controls, never RTL/network evidence."""

import hashlib
import json

import numpy as np
import pytest

from fpga.openeye import network_checkpoint as checkpoint
from fpga.openeye.run_network_inference import Runtime, run_stage_segment
from fpga.openeye.run_residual_block import digest


def int8_hash(a):
    return hashlib.sha256(a.astype("i1").tobytes()).hexdigest()


def stage_record(name, a, scale, output, output_scale):
    return dict(
        layer=name,
        backend="openeye-rtl",
        input_sha256=int8_hash(a),
        input_shape=list(a.shape),
        input_scale=scale,
        accumulator_shape=list(output.shape),
        checked_values=int(output.size),
        accumulator_sha256=digest(output),
        reference_accumulator_sha256=digest(output),
        forwarded_int8_sha256=int8_hash(output),
        output_sha256=digest(output),
        output_scale=output_scale,
    )


def fabricated_state(next_segment):
    count, expected, pending_layout = checkpoint.checkpoint_layout(next_segment)
    a = np.ones((64, 56, 56), dtype=np.int64)
    identity = dict(
        domino_commit="b" * 40,
        chain_id="123",
        input_identity=dict(forwarded_int8_sha256=int8_hash(a)),
    )
    stages, blocks = [], []
    for index, name in enumerate(checkpoint.BLOCKS[:count]):
        group = int(name[5])
        shape = (64 * 2 ** (group - 1), 56 // 2 ** (group - 1), 56 // 2 ** (group - 1))
        first = np.full(shape, 2, dtype=np.int64)
        second = np.full(shape, -3, dtype=np.int64)
        stages.extend(
            [
                stage_record(name + ".conv1", a, 0.125, first, 0.25),
                stage_record(name + ".conv2", first, 0.25, second, 0.5),
            ]
        )
        if name.endswith(".0") and group > 1:
            stages.append(stage_record(name + ".downsample.0", a, 0.125, first, 0.25))
        output = np.full(shape, index + 4, dtype=np.int64)
        blocks.append(
            dict(
                block=name,
                input_sha256=int8_hash(a),
                input_shape=list(a.shape),
                output_shape=list(shape),
                skip_shape=list(shape),
                output_scale=0.125,
                output_sha256=digest(output),
                reference_sha256=digest(output),
                forwarded_int8_sha256=int8_hash(output),
            )
        )
        a = output
    pending = None
    scale = 0.125
    if pending_layout:
        prefix, suffixes = pending_layout
        pending = dict(
            block=prefix,
            original_actual=a.tolist(),
            original_reference=a.tolist(),
            original_scale=scale,
        )
        for suffix in suffixes:
            output = np.full(
                (512, 7, 7), 2 if suffix == "conv1" else -3, dtype=np.int64
            )
            next_scale = 0.25 if suffix == "conv1" else 0.5
            stages.append(
                stage_record(prefix + "." + suffix, a, scale, output, next_scale)
            )
            a, scale = output, next_scale
    assert [s["layer"] for s in stages] == expected
    state = dict(
        schema_version=2,
        completed_segment=next_segment - 1,
        identity=identity,
        stages=stages,
        blocks=blocks,
        actual=a.tolist(),
        reference=a.tolist(),
        scale=scale,
        pending_block=pending,
    )
    return state, identity


@pytest.mark.parametrize("next_segment", range(1, 12))
def test_complete_or_pending_prefix_has_exact_coverage(next_segment):
    state, identity = fabricated_state(next_segment)
    a, ref, scale = checkpoint.validate_state(state, identity, next_segment)
    assert np.array_equal(a, ref) and scale == state["scale"]
    if next_segment == 8:
        assert a.min() == -3  # conv2 BN before residual can be negative.


@pytest.mark.parametrize(
    "fault",
    [
        "legacy-schema",
        "missing-skip",
        "wrong-skip-block",
        "skip-value",
        "skip-oracle",
        "skip-scale",
        "stage-order",
        "stage-input",
        "stage-scale",
        "stage-shape",
        "stage-oracle",
        "missing-stage",
        "actual",
        "reference",
        "scale",
        "wrong-next",
    ],
)
def test_invalid_pending_boundary_rejected_before_next_stage(fault):
    state, identity = fabricated_state(8)
    pending = state["pending_block"]
    if fault == "legacy-schema":
        state["schema_version"] = 1
    elif fault == "missing-skip":
        state["pending_block"] = None
    elif fault == "wrong-skip-block":
        pending["block"] = "layer4.1"
    elif fault == "skip-value":
        pending["original_actual"][0][0][0] += 1
    elif fault == "skip-oracle":
        pending["original_reference"][0][0][0] += 1
    elif fault == "skip-scale":
        pending["original_scale"] = 0.75
    elif fault == "stage-order":
        state["stages"][-2:] = state["stages"][-2:][::-1]
    elif fault == "stage-input":
        state["stages"][-1]["input_sha256"] = "wrong"
    elif fault == "stage-scale":
        state["stages"][-1]["input_scale"] = 0.75
    elif fault == "stage-shape":
        state["stages"][-1]["accumulator_shape"] = [512, 8, 8]
    elif fault == "stage-oracle":
        state["stages"][-1]["reference_accumulator_sha256"] = "wrong"
    elif fault == "missing-stage":
        state["stages"].pop()
    elif fault == "actual":
        state["actual"][0][0][0] += 1
    elif fault == "reference":
        state["reference"][0][0][0] += 1
    elif fault == "scale":
        state["scale"] = 0.75
    with pytest.raises(ValueError):
        checkpoint.validate_state(state, identity, 9 if fault == "wrong-next" else 8)


@pytest.mark.parametrize("next_segment", [7, 8, 10])
def test_pending_skip_and_signed_actual_survive_authenticated_roundtrip(
    tmp_path, monkeypatch, next_segment
):
    # Prefix verifier has independent real-JUnit controls in test_network_evidence.
    # Mock only disk evidence here; tensor/state validation and real AEAD execute.
    monkeypatch.setattr(checkpoint, "validate_prefix_evidence", lambda *a: None)
    state, identity = fabricated_state(next_segment)
    path = tmp_path / "state.json"
    path.write_text(json.dumps(state))
    evidence = tmp_path / "public-evidence"
    evidence.mkdir()
    blob = tmp_path / "boundary.aesgcm"
    key = b"k" * 32
    checkpoint.seal(path, evidence, blob, identity, key)
    restored = tmp_path / "restored"
    checkpoint.unseal(blob, restored, identity, key)
    assert json.loads((restored / "state.json").read_text()) == state
    assert not any("original" in p.name for p in (restored / "evidence").rglob("*"))
    tampered = tmp_path / "tampered.aesgcm"
    raw = bytearray(blob.read_bytes())
    raw[-1] ^= 1
    tampered.write_bytes(raw)
    with pytest.raises(ValueError, match="authentication"):
        checkpoint.unseal(tampered, tmp_path / "rejected", identity, key)


@pytest.mark.parametrize("prefix,start", [("layer4.0", 6), ("layer4.1", 9)])
def test_stage_split_matches_unsplit_residual_and_preserves_actual_original(
    prefix, start, monkeypatch
):
    """Distinct synthetic actual branches catch lost/overwritten skip across jobs."""
    import itertools
    ticks = itertools.count(step=100)
    monkeypatch.setattr('fpga.openeye.runtime_activations.time.perf_counter_ns', lambda: next(ticks))
    original = np.full((256, 14, 14) if start == 6 else (512, 7, 7), 7, dtype=np.int64)

    def engine(calls):
        runtime = object.__new__(Runtime)
        runtime.blocks = []

        def stage(layer, a, ref, scale, bn=None, relu=False, rtl=True):
            assert rtl and np.array_equal(a, ref)
            calls.append((layer, int(a[0, 0, 0]), bn, relu))
            if layer.endswith("downsample.0"):
                assert np.array_equal(a, original) and scale == 0.125
                value, next_scale = 5, 0.25
            elif layer.endswith("conv1"):
                assert np.array_equal(a, original)
                value, next_scale = 2, 0.25
            else:
                assert int(a[0, 0, 0]) == 2 and scale == 0.25
                value, next_scale = -1, 0.25
            out = np.full((512, 7, 7), value, dtype=np.int64)
            return out, out.copy(), next_scale

        runtime.stage = stage
        return runtime

    unsplit_calls = []
    unsplit = engine(unsplit_calls)
    expected, _, expected_scale = unsplit.block(
        prefix, original, original.copy(), 0.125, downsample=start == 6
    )
    calls = []
    split = engine(calls)
    a, ref, scale, pending = original, original.copy(), 0.125, None
    for index in range(start, 9 if start == 6 else 11):
        a, ref, scale, pending = run_stage_segment(split, index, a, ref, scale, pending)
        # Cross a serialization boundary between stage jobs, including scales.
        a = np.asarray(json.loads(json.dumps(a.tolist())), dtype=np.int64)
        ref = np.asarray(json.loads(json.dumps(ref.tolist())), dtype=np.int64)
        pending = json.loads(json.dumps(pending))
        if pending:
            assert not split.blocks
            assert pending["original_actual"] == original.tolist()
    assert pending is None and calls == unsplit_calls
    assert (
        np.array_equal(a, expected)
        and np.array_equal(a, ref)
        and scale == expected_scale
    )
    assert split.blocks == unsplit.blocks


def test_partial_stage_never_appears_to_complete_a_block():
    count, stages, pending = checkpoint.checkpoint_layout(8)
    assert (
        count == 6 and stages[-2:] == ["layer4.0.conv1", "layer4.0.conv2"] and pending
    )
    assert len(checkpoint.checkpoint_layout(11)[1]) == 19
    for name in [
        "network-segment-008-verification.json",
        "network-segment-010-verification.json",
    ]:
        checkpoint.public_path(name)
    for name in [
        "network-state.json",
        "original-actual.json",
        "network-segment-011-verification.json",
    ]:
        with pytest.raises(ValueError):
            checkpoint.public_path(name)


@pytest.mark.parametrize("scale", [float("inf"), -1, 0])
def test_coherently_changed_pending_scales_still_require_positive_finite_values(scale):
    state, identity = fabricated_state(7)
    state["blocks"][-1]["output_scale"] = scale
    state["pending_block"]["original_scale"] = scale
    state["stages"][-1]["input_scale"] = scale
    with pytest.raises(ValueError):
        checkpoint.validate_state(state, identity, 7)


@pytest.mark.parametrize("next_segment", [7, 8, 10])
def test_seal_requires_dut_evidence_for_pending_stage_too(
    tmp_path, monkeypatch, next_segment
):
    import fpga.openeye.verify_network_evidence as verifier

    state, identity = fabricated_state(next_segment)
    identity.update(
        checkpoint_sha256="a" * 64,
        scheme_sha256="b" * 64,
        packed_manifest_sha256="c" * 64,
    )
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    for stage in state["stages"]:
        name = stage["layer"].replace(".", "-")
        source = dict(
            source_layer=stage["layer"],
            source_checkpoint_sha256="a" * 64,
            source_scheme_sha256="b" * 64,
            source_packed_manifest_sha256="c" * 64,
        )
        source_path = evidence / f"{name}-fixture-manifest.json"
        source_path.write_text(json.dumps(source))
        (evidence / f"{name}-stage-verification.json").write_text(json.dumps(stage))
        root = evidence / name
        root.mkdir()
        (root / "regression_manifest.txt").write_text(
            "fixture_manifest_sha256="
            + hashlib.sha256(source_path.read_bytes()).hexdigest()
            + "\n"
        )
    seen = []

    def verify_layer(root, stage):
        seen.append(stage["layer"])
        if stage["layer"] == state["stages"][-1]["layer"]:
            raise ValueError("Missing real DUT JUnit for pending stage")

    monkeypatch.setattr(verifier, "verify_layer", verify_layer)
    path = tmp_path / "state.json"
    path.write_text(json.dumps(state))
    blob = tmp_path / "rejected.aesgcm"
    with pytest.raises(ValueError, match="Missing real DUT JUnit"):
        checkpoint.seal(path, evidence, blob, identity, b"k" * 32)
    assert seen == [s["layer"] for s in state["stages"]]
    assert not blob.exists()


def test_execute_resumes_each_layer4_stage_from_encrypted_actual_state(
    tmp_path, monkeypatch
):
    """Fake DUT covers CLI orchestration + real AEAD, never a hardware claim."""
    import base64
    from types import SimpleNamespace
    import fpga.openeye.run_network_inference as runner

    state, identity = fabricated_state(6)
    ingress = dict(verified_utc="first-job", method="fabricated-test")
    bundle = tmp_path / "bundle"
    packed = bundle / "mac20/packed-v1"
    packed.mkdir(parents=True)
    (packed / "manifest.json").write_text("{}")
    evidence = tmp_path / "prefix-evidence"
    evidence.mkdir()
    (evidence / "ingress-verification.json").write_text(json.dumps(ingress))
    path = tmp_path / "prefix-state.json"
    path.write_text(json.dumps(state))
    monkeypatch.setattr(checkpoint, "validate_prefix_evidence", lambda *a: None)
    monkeypatch.setattr(
        runner, "verify_bundle", lambda _: dict(ingress, verified_utc="fresh-job")
    )
    monkeypatch.setattr(
        runner,
        "source_activation",
        lambda *a: (
            np.zeros((64, 56, 56), dtype=np.int64),
            99,
            identity["input_identity"],
        ),
    )
    monkeypatch.setattr(runner, "resume_identity", lambda *a: identity)
    monkeypatch.setattr(
        runner.subprocess,
        "check_output",
        lambda command, **kw: (
            runner.EXPECTED_OPENEYE_COMMIT + "\n" if "rev-parse" in command else ""
        ),
    )
    key = b"k" * 32
    monkeypatch.setenv("OPENEYE_NETWORK_RESUME_KEY_B64", base64.b64encode(key).decode())
    blob = tmp_path / "prefix.aesgcm"
    checkpoint.seal(path, evidence, blob, identity, key)
    calls = []

    class FakeRuntime(Runtime):
        def stage(self, layer, a, ref, scale, bn=None, relu=False, rtl=True):
            assert rtl and not self.reference_only and np.array_equal(a, ref)
            if layer == "layer4.0.downsample.0":
                assert (
                    a.shape == (256, 14, 14) and int(a[0, 0, 0]) == 9 and scale == 0.125
                )
            elif layer == "layer4.0.conv1":
                assert a.shape == (256, 14, 14) and int(a[0, 0, 0]) == 9
            else:
                assert a.shape == (512, 7, 7)
            value, next_scale = (
                (2, 0.25)
                if layer.endswith("conv1")
                else (-1, 0.25) if layer.endswith("conv2") else (9, 0.25)
            )
            output = np.full((512, 7, 7), value, dtype=np.int64)
            self.stages.append(stage_record(layer, a, scale, output, next_scale))
            calls.append(layer)
            return output, output.copy(), next_scale

    monkeypatch.setattr(runner, "Runtime", FakeRuntime)
    for index in range(6, 11):
        private = tmp_path / f"private-{index}"
        evidence = tmp_path / f"evidence-{index}"
        args = SimpleNamespace(
            bundle_root=bundle,
            openeye_root=tmp_path / "OpenEye",
            private_root=private,
            evidence=evidence,
            mode="full-network",
            spatial_size=56,
            sample=0,
            reference_only=False,
            segment_index=index,
            chain_id="123",
            resume_checkpoint=blob,
            resume_context_only=None,
        )
        result = runner.execute(args)
        state = json.loads((private / "network-state.json").read_text())
        count, expected, pending = checkpoint.checkpoint_layout(index + 1)
        assert (
            len(state["blocks"]) == count
            and [s["layer"] for s in state["stages"]] == expected
        )
        assert (
            result["result"] == "SEGMENT_PASS" and not result["full_network_complete"]
        )
        assert result["checkpoint_boundary"] == (
            "checked-stage" if pending else "checked-block"
        )
        assert not (evidence / "network-verification.json").exists()
        blob = tmp_path / f"segment-{index}.aesgcm"
        checkpoint.seal(private / "network-state.json", evidence, blob, identity, key)
    assert calls == [
        "layer4.0.conv1",
        "layer4.0.conv2",
        "layer4.0.downsample.0",
        "layer4.1.conv1",
        "layer4.1.conv2",
    ]
