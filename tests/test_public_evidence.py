"""Reject public tensor payloads; preserve counters and real failure status."""

import hashlib
import json
import xml.etree.ElementTree as ET

import pytest

from fpga.openeye.publish_public_evidence import counter_log, publish
from fpga.openeye.summarize_wrapper_metrics import parse_log


def test_simulator_payload_removed_and_counter_semantics_preserved(tmp_path):
    source = tmp_path / "private"
    source.mkdir()
    log = source / "wrapper-conv-packed-layer.log"
    log.write_text(
        "layer-0 input activations (first 34): [41, 42, 43]\n"
        "private token should not be public\n"
        "Host will send: iact 4, wght 2, psum 0, quantize 0 words\n"
        'DOMINO_HW_METRICS {"schema_version":2,"total_cycles":10,"state_cycles":{"GET_WGHT":2}}\n'
    )
    before = parse_log(log)
    output = tmp_path / "public"
    publish(source, output)
    assert parse_log(output / log.name) == before
    text = (output / log.name).read_text()
    assert "activation" not in text and "private token" not in text
    line = (output / "SHA256SUMS").read_text().splitlines()[0]
    sha, name = line.split(maxsplit=1)
    assert hashlib.sha256((output / name).read_bytes()).hexdigest() == sha
    assert log.read_text().startswith("layer-0 input")  # Never mutate private source.


@pytest.mark.parametrize("tag", ["failure", "error", "skipped"])
def test_junit_failure_never_becomes_pass_and_payload_is_removed(tmp_path, tag):
    source = tmp_path / "private"
    source.mkdir()
    path = source / "junit-0000.xml"
    path.write_text(
        f'<testsuites><testsuite tests="1" failures="1"><testcase classname="OpenEye_FPGA_tb" name="start_test_fpga"><{tag} message="SECRET VALUES">SECRET TENSOR</{tag}><system-out>SECRET TOKEN</system-out></testcase></testsuite></testsuites>'
    )
    output = tmp_path / "public"
    publish(source, output)
    text = (output / path.name).read_text()
    assert "SECRET" not in text
    assert len(list(ET.fromstring(text).iter(tag))) == 1
    assert list(ET.fromstring(text).iter("testsuite"))[0].get("failures") == "1"


@pytest.mark.parametrize(
    "key",
    [
        "weights",
        "actual",
        "reference",
        "original_actual",
        "original_reference",
        "coefficients",
        "token",
        "refresh_token",
        "access_token",
    ],
)
def test_private_json_keys_cannot_be_published(tmp_path, key):
    source = tmp_path / "private"
    source.mkdir()
    (source / "network-verification.json").write_text(
        json.dumps({"nested": {key: [1, 2, 3]}})
    )
    with pytest.raises(ValueError, match="Private"):
        publish(source, tmp_path / "public")


@pytest.mark.parametrize(
    "path",
    ["network-state.json", "network-output.bin", "checkpoint.aesgcm", "rclone.conf"],
)
def test_private_paths_rejected_even_if_parent_is_evidence(tmp_path, path):
    source = tmp_path / "private"
    source.mkdir()
    (source / path).write_bytes(b"private")
    with pytest.raises(ValueError):
        publish(source, tmp_path / "public")


def test_metrics_cannot_smuggle_text_or_tensor_arrays(tmp_path):
    path = tmp_path / "log"
    for value in ['"secret"', "[1,2,3]"]:
        path.write_text('DOMINO_HW_METRICS {"extra":' + value + "}\n")
        with pytest.raises(ValueError):
            counter_log(path)


def test_publish_requires_disjoint_new_copy(tmp_path):
    source = tmp_path / "private"
    source.mkdir()
    with pytest.raises(ValueError):
        publish(source, source / "child")
    output = tmp_path / "public"
    output.mkdir()
    with pytest.raises(ValueError):
        publish(source, output)


def test_host_operation_metadata_survives_publication_without_tensor_fields(tmp_path):
    source = tmp_path / 'private'
    source.mkdir()
    metadata = dict(host_operations=dict(
        forward_path=[dict(operation='batchnorm_affine_requantization', elements=128, calls=1, wall_ns=1234)],
        oracle_path=[dict(operation='relu_int8_clamp_cast', elements=128, calls=1, wall_ns=5678)]))
    (source / 'network-verification.json').write_text(json.dumps(metadata))
    output = tmp_path / 'public'
    publish(source, output)
    assert json.loads((output / 'network-verification.json').read_text()) == metadata
