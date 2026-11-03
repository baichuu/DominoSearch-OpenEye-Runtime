#!/usr/bin/env python3
"""Validate network dispatch identity before paying for simulation gates.

This is an execution plan, never RTL or checkpoint correctness evidence.
"""
import argparse
import json
from pathlib import Path
import re

NETWORK_SCOPES = ('hybrid-network', 'hybrid-network-resumable')
RUNTIME_SCOPES = ('residual-block-pilot', 'downsample-block-pilot',
                  'network-fc-smoke', *NETWORK_SCOPES)
PRIVATE_REMOTE = 'domino-drive:DominoSearch-runtime-private'


def plan(scope, architecture, spatial_size, commit, pin, resume_run_id='', resume_metadata=None,
         private_remote=PRIVATE_REMOTE, private_policy='owner-only'):
    if scope not in RUNTIME_SCOPES:
        raise ValueError('Unknown runtime scope')
    if not re.fullmatch(r'[0-9a-f]{40}', commit) or not re.fullmatch(r'[0-9a-f]{40}', pin):
        raise ValueError('Require full execution commit and OpenEye pin')
    if scope in NETWORK_SCOPES and (architecture != 'full-layer-r3' or spatial_size != '56'):
        raise ValueError('Full network requires architecture=full-layer-r3 and spatial_size=56')
    if scope == 'hybrid-network-resumable' and (private_remote != PRIVATE_REMOTE or private_policy != 'owner-only'):
        raise ValueError('Resumable network requires the approved private root and owner-only policy')
    if resume_run_id:
        if scope != 'hybrid-network-resumable' or not re.fullmatch(r'[1-9][0-9]*', resume_run_id):
            raise ValueError('Resume run ID requires resumable network scope and a positive numeric ID')
        if not isinstance(resume_metadata, dict):
            raise ValueError('Require original Actions run metadata before attempting resume')
        if (resume_metadata.get('id') != int(resume_run_id)
                or resume_metadata.get('head_sha') != commit
                or resume_metadata.get('path') != '.github/workflows/openeye-full-layer.yml'
                or resume_metadata.get('event') != 'workflow_dispatch'
                or resume_metadata.get('status') != 'completed'):
            raise ValueError('Resume requires a completed manual run of this workflow at the same execution commit; start a fresh chain after code/pin changes')
    elif resume_metadata is not None:
        raise ValueError('Unexpected resume metadata for a fresh chain')
    return dict(schema_version=1, result='PREFLIGHT_PLAN_ONLY', scope=scope,
                domino_commit=commit, openeye_commit=pin,
                full_network=scope in NETWORK_SCOPES,
                architecture=architecture if scope in NETWORK_SCOPES else 'scope-specific',
                spatial_size=56 if scope in NETWORK_SCOPES else 'scope-specific',
                resume_run_id=resume_run_id or None,
                requires_same_dispatch_gates=scope in NETWORK_SCOPES,
                checkpoint_reuse='authenticated completed blocks/stages, original skip retained; same execution identity only',
                rtl_executed=False, full_network_complete=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope', required=True, choices=RUNTIME_SCOPES)
    parser.add_argument('--architecture', required=True)
    parser.add_argument('--spatial-size', required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--pin', required=True)
    parser.add_argument('--resume-run-id', default='')
    parser.add_argument('--resume-metadata', type=Path)
    parser.add_argument('--private-remote', default=PRIVATE_REMOTE)
    parser.add_argument('--private-policy', default='owner-only')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads(args.resume_metadata.read_text()) if args.resume_metadata else None
    result = plan(args.scope, args.architecture, args.spatial_size, args.commit,
                  args.pin, args.resume_run_id, metadata, args.private_remote, args.private_policy)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
