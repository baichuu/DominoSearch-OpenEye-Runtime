import base64
import configparser
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from fpga.openeye.prepare_ci_rclone_config import prepare_config


def encoded_config(scope="drive.readonly", refresh=True):
    token = {"access_token": "test-access", "expiry": "2099-01-01T00:00:00Z"}
    if refresh:
        token["refresh_token"] = "test-refresh"
    text = f"[domino-drive]\ntype = drive\nscope = {scope}\nclient_id = test-client\nclient_secret = test-secret\ntoken = {json.dumps(token)}\n"
    return base64.b64encode(text.encode()).decode()


def test_requires_refresh_without_altering_existing_grant():
    result = prepare_config(encoded_config())
    config = configparser.ConfigParser()
    config.read_string(result)
    remote = config["domino-drive"]
    assert remote["scope"] == "drive.readonly"
    assert remote["client_id"] == "test-client"
    assert remote["client_secret"] == "test-secret"
    assert json.loads(remote["token"]) == {
        "access_token": "test-access", "refresh_token": "test-refresh",
        "expiry": "1970-01-01T00:00:00Z"}


def test_accepts_wrapped_base64_from_standard_cli():
    encoded = encoded_config()
    wrapped = "\n".join(encoded[i:i + 76] for i in range(0, len(encoded), 76)) + "\n"
    assert prepare_config(wrapped) == prepare_config(encoded)


@pytest.mark.parametrize("value", ["", "%%%", base64.b64encode(b"bad-config-test-secret").decode(), encoded_config(refresh=False)])
def test_rejects_invalid_input_without_exposing_credentials(value):
    with pytest.raises(ValueError) as error:
        prepare_config(value)
    assert "test-secret" not in str(error.value)
    assert "test-access" not in str(error.value)


def test_existing_grant_scope_is_preserved_not_expanded():
    config = configparser.ConfigParser()
    config.read_string(prepare_config(encoded_config(scope="drive")))
    assert config["domino-drive"]["scope"] == "drive"


def test_rejects_other_backend():
    text = base64.b64decode(encoded_config()).decode().replace("type = drive", "type = s3")
    with pytest.raises(ValueError, match="Google Drive"):
        prepare_config(base64.b64encode(text.encode()).decode())


def test_rejects_unrelated_remotes():
    text = base64.b64decode(encoded_config()).decode() + "[other]\ntype = drive\n"
    with pytest.raises(ValueError, match="only the domino-drive"):
        prepare_config(base64.b64encode(text.encode()).decode())


def test_cli_private_exclusive_creation(tmp_path):
    output = tmp_path / "rclone.conf"
    env = {**os.environ, "DOMINO_RCLONE_CONFIG_B64": encoded_config()}
    script = Path(__file__).resolve().parents[1] / "fpga/openeye/prepare_ci_rclone_config.py"
    command = [sys.executable, str(script), "--output", str(output)]
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    assert result.returncode == 0
    assert output.stat().st_mode & 0o777 == 0o600
    before = output.read_bytes()
    rerun = subprocess.run(command, env=env, capture_output=True, text=True)
    assert rerun.returncode == 2
    assert output.read_bytes() == before
    assert "test-access" not in result.stdout + result.stderr + rerun.stdout + rerun.stderr
