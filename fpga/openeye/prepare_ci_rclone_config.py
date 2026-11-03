#!/usr/bin/env python3
"""Prepare a private runner-only Drive config without exposing credentials."""

import argparse
import base64
import configparser
import io
import json
import os
from pathlib import Path
import sys


def prepare_config(encoded: str) -> str:
    if not encoded:
        raise ValueError("Missing repository secret DOMINO_RCLONE_CONFIG_B64")
    config = configparser.ConfigParser(interpolation=None)
    try:
        config.read_string(base64.b64decode("".join(encoded.split()), validate=True).decode("utf-8"))
    except (ValueError, UnicodeError, configparser.Error):
        raise ValueError("Invalid encoded rclone configuration") from None
    if config.defaults() or config.sections() != ["domino-drive"]:
        raise ValueError("CI config must contain only the domino-drive remote")
    remote = config["domino-drive"]
    if remote.get("type") != "drive":
        raise ValueError("CI requires a Google Drive remote")
    try:
        token = json.loads(remote.get("token", ""))
    except ValueError:
        raise ValueError("Invalid Drive token metadata") from None
    if not isinstance(token, dict) or not token.get("refresh_token"):
        raise ValueError("CI Drive access requires a refresh token")
    # A skewed local clock can store an already-invalid access token with a
    # future expiry. Force the runner's first rclone command to refresh using
    # the existing grant. Never alter the secret, scopes, or host clock.
    token["expiry"] = "1970-01-01T00:00:00Z"
    remote["token"] = json.dumps(token, separators=(",", ":"))
    output = io.StringIO()
    config.write(output)
    return output.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        content = prepare_config(os.environ.get("DOMINO_RCLONE_CONFIG_B64", ""))
        # Exclusive creation prevents overwriting a credential file. Mode is
        # private from its creation, rather than chmod after writing it.
        fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(content)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2
    except OSError:
        print("Cannot create private runner rclone configuration", file=sys.stderr)
        return 2
    print("Prepared private Drive config; existing scope preserved; refresh required")
    return 0


if __name__ == "__main__":
    sys.exit(main())
