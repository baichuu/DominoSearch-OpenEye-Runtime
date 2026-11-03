#!/usr/bin/env python3
"""Select latest same-run gate artifact, including rerun-failed-job attempts."""
import argparse
from pathlib import Path
import re
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fpga.openeye.verify_network_evidence import verify


def select(container, output, mode, commit):
    gate = 'downsample-block-pilot' if mode == 'downsample-block' else 'network-fc-smoke'
    pattern = re.compile(r'openeye-runtime-' + gate + '-' + re.escape(commit) + r'-(\d+)')
    choices = [(int(m.group(1)), p) for p in container.iterdir()
               if (m := pattern.fullmatch(p.name)) is not None and p.is_dir()]
    if not choices or output.exists():
        raise ValueError('Require complete gate artifact and new selected directory')
    attempt, source = max(choices)
    # Never fall back to an older green artifact if the latest one is bad.
    result = verify(source, mode, commit)
    shutil.copytree(source, output)
    return dict(result, gate_attempt=attempt)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--container', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=('downsample-block', 'fc-smoke'), required=True)
    p.add_argument('--domino-commit', required=True)
    args = p.parse_args()
    import json
    print(json.dumps(select(args.container, args.output, args.mode, args.domino_commit), sort_keys=True))


if __name__ == '__main__':
    main()
