#!/usr/bin/env python3
"""Authenticated private network boundaries; never an Actions tensor artifact.

Only completed checked boundaries are reusable. Layer4 also preserves the
original skip across checked convolution stages. No incomplete tile is reused.
"""
from __future__ import annotations

import argparse
import base64
import configparser
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib

# CI invokes this file by path from outside the repository. Resolve sibling
# package imports from the script location, not cwd or an inherited PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

MAGIC = b'DOMINO-NETWORK-STATE-v2\0'
MAX_BYTES = 256 * 1024 * 1024
BLOCKS = tuple(f'layer{g}.{i}' for g in range(1, 5) for i in range(2))
FINAL_SEGMENT = 11
STAGE_SEGMENTS = {6: ('layer4.0', 'conv1'), 7: ('layer4.0', 'conv2'),
                  8: ('layer4.0', 'downsample.0'),
                  9: ('layer4.1', 'conv1'), 10: ('layer4.1', 'conv2')}
PRIVATE_REMOTES = ('domino-drive:DominoSearch-runtime-private', 'domino-drive:DominoSearch-artifacts/runtime-private')
TRANSIENT_DRIVE_403 = frozenset(('rateLimitExceeded', 'userRateLimitExceeded'))
DRIVE_ACL_MAX_ATTEMPTS = 8
DRIVE_ACL_MAX_BACKOFF_SECONDS = 32
SAFE_DRIVE_REASONS = TRANSIENT_DRIVE_403 | frozenset((
    'dailyLimitExceeded', 'insufficientPermissions', 'forbidden', 'domainPolicy',
    'appNotAuthorizedToFile', 'authError', 'notFound', 'backendError',
    'accessNotConfigured', 'SERVICE_DISABLED', 'ACCESS_TOKEN_SCOPE_INSUFFICIENT',
    'PERMISSION_DENIED', 'UNAUTHENTICATED',
))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def public_path(value):
    """Allow only existing correctness metadata/log/JUnit names, not tensors."""
    parts = value.split('/')
    if any(p in ('', '.', '..') for p in parts) or '\\' in value:
        raise ValueError('Unsafe checkpoint evidence path')
    layer_name = r'(?:layer[1-4]-[01]-(?:conv[12]|downsample-0)|fc|focused-test_network_channel_tiles_smo\d+)'
    if len(parts) > 1 and not re.fullmatch(layer_name, parts[0]):
        raise ValueError('Unknown/private evidence directory')
    if len(parts) > 3 or (len(parts) == 3 and parts[1] != 'spatial-tiles'):
        raise ValueError('Unexpected nested evidence directory')
    allowed = (parts[-1] in {
        'SHA256SUMS', 'regression_manifest.txt', 'hardware_metrics.json',
        'spatial_layer_verification.json', 'wrapper-conv-packed-layer.log',
        'ingress-verification.json', 'network-verification.json',
    } or re.fullmatch(r'(?:layer[1-4]-[01]-(?:conv[12]|downsample-0)|fc)-(?:fixture-manifest|stage-verification)\.json', parts[-1])
       or re.fullmatch(r'network-segment-(?:00[0-9]|010)-verification\.json', parts[-1])
       or re.fullmatch(r'(?:tile-\d{4}\.json|junit-\d{4}\.xml)', parts[-1]))
    if not allowed:
        raise ValueError('Private/unknown payload in public evidence')
    return Path(*parts)


def stages_through(block_count):
    result = []
    for block in BLOCKS[:block_count]:
        result.extend((block + '.conv1', block + '.conv2'))
        if block.endswith('.0') and not block.startswith('layer1.'):
            result.append(block + '.downsample.0')
    return result


def checkpoint_layout(next_segment):
    """Expected complete blocks plus the optional checked layer4 main branch."""
    if not isinstance(next_segment, int) or isinstance(next_segment, bool) or not 1 <= next_segment <= FINAL_SEGMENT:
        raise ValueError('Wrong checkpoint segment order')
    count = next_segment if next_segment <= 6 else 6 if next_segment <= 8 else 7 if next_segment <= 10 else 8
    pending = {7: ('layer4.0', ('conv1',)),
               8: ('layer4.0', ('conv1', 'conv2')),
               10: ('layer4.1', ('conv1',))}.get(next_segment)
    stages = stages_through(count)
    if pending:
        stages += [pending[0] + '.' + name for name in pending[1]]
    return count, stages, pending


def checked_boundary(actual, reference, shape, minimum=0):
    import numpy as np
    a, ref = np.asarray(actual), np.asarray(reference)
    if a.dtype.kind not in 'iu' or ref.dtype.kind not in 'iu':
        raise ValueError('Checkpoint boundary must contain integer activations')
    if (a.shape != shape or ref.shape != shape or not np.array_equal(a, ref)
            or a.min() < minimum or a.max() > 127):
        raise ValueError('Invalid DUT/reference checkpoint boundary')
    return a.astype(np.int64), ref.astype(np.int64)


def validate_state(state, identity, next_segment):
    import numpy as np
    count, expected_stages, pending_layout = checkpoint_layout(next_segment)
    if (state.get('schema_version') != 2 or state.get('identity') != identity
            or state.get('completed_segment') != next_segment - 1):
        raise ValueError('Wrong checkpoint identity or segment order')
    stages, blocks = state['stages'], state['blocks']
    if ([s['layer'] for s in stages] != expected_stages
            or any(s['backend'] != 'openeye-rtl' for s in stages)
            or [b['block'] for b in blocks] != list(BLOCKS[:count])):
        raise ValueError('Checkpoint lacks complete RTL prefix')
    group = (count - 1) // 2 + 1
    expected = (64 * 2 ** (group - 1), 56 // 2 ** (group - 1), 56 // 2 ** (group - 1))
    from fpga.openeye.run_residual_block import digest
    forwarded = identity['input_identity']['forwarded_int8_sha256']
    for block in blocks:
        if (block['input_sha256'] != forwarded or block['output_sha256'] != block['reference_sha256']
                or not np.isfinite(block['output_scale']) or block['output_scale'] <= 0):
            raise ValueError('Broken checkpoint activation chain')
        selected = {s['layer']: s for s in stages if s['layer'].startswith(block['block'] + '.')}
        conv1, conv2 = selected[block['block'] + '.conv1'], selected[block['block'] + '.conv2']
        if (conv1['input_sha256'] != forwarded
                or conv2['input_sha256'] != conv1['forwarded_int8_sha256']
                or conv1['input_shape'] != block['input_shape']
                or conv2['accumulator_shape'] != block['output_shape']):
            raise ValueError('Broken checkpoint stage activation/geometry chain')
        projection = selected.get(block['block'] + '.downsample.0')
        if projection is not None and (projection['input_sha256'] != forwarded
                                      or projection['accumulator_shape'] != block['skip_shape']):
            raise ValueError('Checkpoint projection changed original skip')
        forwarded = block['forwarded_int8_sha256']
    pending = state.get('pending_block')
    if pending_layout:
        prefix, suffixes = pending_layout
        if not isinstance(pending, dict) or set(pending) != {'block', 'original_actual', 'original_reference', 'original_scale'} or pending['block'] != prefix:
            raise ValueError('Missing/wrong pending original skip')
        original, _ = checked_boundary(pending['original_actual'], pending['original_reference'], expected)
        if (hashlib.sha256(original.astype('i1').tobytes()).hexdigest() != forwarded
                or digest(original) != blocks[-1]['output_sha256']
                or pending['original_scale'] != blocks[-1]['output_scale']):
            raise ValueError('Pending skip differs from authenticated block output')
        selected = stages[-len(suffixes):]
        previous_shape, previous_scale = list(expected), pending['original_scale']
        for stage in selected:
            if (stage['input_sha256'] != forwarded or stage['input_shape'] != previous_shape
                    or stage['input_scale'] != previous_scale
                    or stage['accumulator_shape'] != [512, 7, 7]
                    or stage['checked_values'] != 512 * 7 * 7
                    or not np.isfinite(stage['output_scale']) or stage['output_scale'] <= 0
                    or stage['accumulator_sha256'] != stage['reference_accumulator_sha256']):
                raise ValueError('Broken pending-stage activation/geometry chain')
            forwarded = stage['forwarded_int8_sha256']
            previous_shape, previous_scale = [512, 7, 7], stage['output_scale']
        a, ref = checked_boundary(state['actual'], state['reference'], (512, 7, 7), -127 if len(suffixes) == 2 else 0)
        expected_digest, expected_scale = stages[-1]['output_sha256'], stages[-1]['output_scale']
    else:
        if pending is not None:
            raise ValueError('Unexpected pending skip at completed block')
        a, ref = checked_boundary(state['actual'], state['reference'], expected)
        expected_digest, expected_scale = blocks[-1]['output_sha256'], blocks[-1]['output_scale']
    if (hashlib.sha256(a.astype('i1').tobytes()).hexdigest() != forwarded
            or digest(a) != expected_digest
            or state['scale'] != expected_scale or not np.isfinite(state['scale']) or state['scale'] <= 0):
        raise ValueError('Checkpoint tensor/scale does not match checked DUT evidence')
    return a, ref, state['scale']


def validate_prefix_evidence(evidence, state):
    from fpga.openeye.verify_network_evidence import verify_layer
    from fpga.openeye.prepare_packed_layer_fixture import load_json
    from fpga.openeye.openeye_evidence_bundle import parse_key_values, sha256
    for stage in state['stages']:
        name = stage['layer'].replace('.', '-')
        if load_json(evidence / f'{name}-stage-verification.json') != stage:
            raise ValueError('Checkpoint stage metadata differs from evidence')
        source = load_json(evidence / f'{name}-fixture-manifest.json')
        if (source['source_layer'] != stage['layer']
                or source['source_checkpoint_sha256'] != state['identity']['checkpoint_sha256']
                or source['source_scheme_sha256'] != state['identity']['scheme_sha256']
                or source['source_packed_manifest_sha256'] != state['identity']['packed_manifest_sha256']):
            raise ValueError('Wrong checkpoint prefix source')
        if parse_key_values(evidence / name / 'regression_manifest.txt')['fixture_manifest_sha256'] != sha256(evidence / f'{name}-fixture-manifest.json'):
            raise ValueError('Checkpoint RTL did not consume the declared source fixture')
        verify_layer(evidence / name, stage)


def key_from_environment():
    try:
        key = base64.b64decode(os.environ['OPENEYE_NETWORK_RESUME_KEY_B64'], validate=True)
        if len(key) != 32:
            raise ValueError()
    except (KeyError, ValueError) as exc:
        raise ValueError('Require dedicated 32-byte network resume secret') from exc
    return key


def seal(state_path, evidence, output, context, key):
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    if output.exists() or state_path.is_symlink() or evidence.is_symlink():
        raise ValueError('Require regular state and new ciphertext destination')
    state = json.loads(state_path.read_text())
    if state['identity'] != context:
        raise ValueError('Wrong seal context')
    validate_state(state, context, state['completed_segment'] + 1)
    validate_prefix_evidence(evidence, state)
    files = {}
    for path in sorted(evidence.rglob('*')):
        if path.is_symlink():
            raise ValueError('Symlink in evidence')
        if path.is_file():
            relative = str(path.relative_to(evidence))
            if relative == 'SHA256SUMS':
                continue  # Prefix checksum snapshots stay in prior public artifacts.
            public_path(relative)
            files[relative] = base64.b64encode(path.read_bytes()).decode()
    payload = canonical(dict(state=state, evidence=files))
    if len(payload) > MAX_BYTES:
        raise ValueError('Checkpoint exceeds size cap')
    nonce = os.urandom(12)
    encrypted = MAGIC + nonce + AESGCM(key).encrypt(nonce, zlib.compress(payload), MAGIC + canonical(context))
    output.parent.mkdir(parents=True, exist_ok=True)
    with os.fdopen(os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'wb') as stream:
        stream.write(encrypted)
    return hashlib.sha256(encrypted).hexdigest()


def unseal(source, destination, context, key):
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.exceptions import InvalidTag
    if source.is_symlink() or destination.exists() or source.stat().st_size > MAX_BYTES:
        raise ValueError('Unsafe restore input/destination')
    raw = source.read_bytes()
    if not raw.startswith(MAGIC):
        raise ValueError('Wrong checkpoint format')
    nonce = raw[len(MAGIC):len(MAGIC) + 12]
    try:
        packed = AESGCM(key).decrypt(nonce, raw[len(MAGIC) + 12:], MAGIC + canonical(context))
        decoder = zlib.decompressobj()
        payload = decoder.decompress(packed, MAX_BYTES + 1)
        if len(payload) > MAX_BYTES or not decoder.eof or decoder.unused_data:
            raise ValueError('Invalid/oversized compressed checkpoint')
        document = json.loads(payload)
    except (InvalidTag, zlib.error, json.JSONDecodeError) as exc:
        raise ValueError('Checkpoint authentication/format failed') from exc
    state = document['state']
    validate_state(state, context, state['completed_segment'] + 1)
    files = [(public_path(name), base64.b64decode(value, validate=True)) for name, value in document['evidence'].items()]
    destination.mkdir(parents=True, mode=0o700)
    evidence = destination / 'evidence'
    evidence.mkdir()
    for relative, value in files:
        path = evidence / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(value)
    with (destination / 'state.json').open('x') as stream:
        json.dump(state, stream)
    validate_prefix_evidence(evidence, state)
    return hashlib.sha256(raw).hexdigest()


def writable_drive_config(path):
    config = configparser.ConfigParser(interpolation=None)
    config.read(path)
    if (config.sections() != ['domino-drive'] or config['domino-drive'].get('type') != 'drive'
            or config['domino-drive'].get('scope') not in ('drive', 'drive.file')):
        raise ValueError('Private resume needs existing write grant; never expand OAuth scope automatically')


def validate_sharing(permissions, policy):
    classes = sorted((p['type'],p['role']) for p in permissions)
    expected = [('user','owner')]
    if policy == 'restricted-users':
        expected.append(('user','writer'))
    elif policy != 'owner-only':
        raise ValueError('Unknown checkpoint sharing policy')
    if classes != sorted(expected):
        raise ValueError('Checkpoint folder ACL differs from user-approved privacy policy; no sharing changes made')


def drive_error_reasons(error):
    """Log only known reason codes; never provider messages, URLs or tokens."""
    try:
        document = json.loads(error.read(8192))
        payload = document.get('error', {}) if isinstance(document, dict) else {}
        if not isinstance(payload, dict):
            return ('unknown',)
        records = payload.get('errors') or payload.get('details') or []
        if not isinstance(records, list):
            return ('unknown',)
        raw = [row.get('reason') if isinstance(row, dict) else None for row in records]
        return tuple(sorted({r if isinstance(r, str) and r in SAFE_DRIVE_REASONS else 'unknown'
                             for r in raw})) or ('unknown',)
    except (ValueError, OSError):
        return ('unknown',)
    finally:
        error.close()


def drive_acl_page(request):
    """Bounded read-only retry; permanent/unknown 403 errors remain fatal."""
    for attempt in range(DRIVE_ACL_MAX_ATTEMPTS):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                try:
                    return json.load(response)
                except (ValueError, UnicodeError):
                    raise ValueError('Malformed Drive ACL response; access refused') from None
        except urllib.error.HTTPError as error:
            status, reasons = error.code, drive_error_reasons(error)
            retry = (status in (429, 500, 502, 503, 504)
                     or (status == 403 and set(reasons) <= TRANSIENT_DRIVE_403))
        except (urllib.error.URLError, TimeoutError):
            status, reasons, retry = None, ('transportError',), True
        again = retry and attempt + 1 < DRIVE_ACL_MAX_ATTEMPTS
        print(json.dumps(dict(event='drive-acl-error', http_status=status, reasons=reasons,
                              attempt=attempt+1, max_attempts=DRIVE_ACL_MAX_ATTEMPTS, retry=again)), file=sys.stderr)
        if not again:
            raise ValueError('Drive ACL request failed: status='+str(status)+
                             ', reasons='+','.join(reasons)+'; checkpoint access refused') from None
        # Cross a minute-scale throttle window without unbounded retry or
        # retrying access-denied/unknown errors. Each sleep stays below 33s.
        time.sleep(min(2**attempt, DRIVE_ACL_MAX_BACKOFF_SECONDS) + random.random())


def drive_permissions(folder_id, token):
    """Require complete pagination; no owner-only claim from a partial ACL."""
    base = 'https://www.googleapis.com/drive/v3/files/'+urllib.parse.quote(folder_id, safe='')+'/permissions'
    permissions, seen = [], set()
    page_token = None
    for _ in range(10):
        params = dict(fields='nextPageToken,permissions(type,role)', pageSize='100')
        if page_token is not None:
            params['pageToken'] = page_token
        request = urllib.request.Request(base+'?'+urllib.parse.urlencode(params),
                                         headers={'Authorization': 'Bearer '+token})
        page = drive_acl_page(request)
        if (not isinstance(page, dict) or not isinstance(page.get('permissions'), list)
                or any(not isinstance(row, dict) or not isinstance(row.get('type'), str)
                       or not isinstance(row.get('role'), str) for row in page['permissions'])):
            raise ValueError('Malformed Drive ACL page; access refused')
        permissions.extend(page['permissions'])
        page_token = page.get('nextPageToken')
        if page_token is None or page_token == '':
            return permissions
        if not isinstance(page_token, str) or page_token in seen:
            raise ValueError('Invalid Drive ACL pagination; access refused')
        seen.add(page_token)
    raise ValueError('Drive ACL pagination limit exceeded; access refused')


def verify_private_drive(config_path, remote, policy):
    if remote not in PRIVATE_REMOTES:
        raise ValueError('Private checkpoint remote outside authorized roots')
    parent,name = remote.rsplit('/',1) if '/' in remote else (remote.split(':')[0]+':',remote.split(':')[1])
    # Parent listing gives the real backend folder ID; stat of a remote root
    # does not. Refresh with the existing grant, then read token privately.
    rows = json.loads(subprocess.check_output(['rclone','lsjson','--dirs-only','--config',str(config_path),parent],text=True))
    matches = [p for p in rows if p['Name']==name]
    if len(matches)!=1 or not matches[0].get('ID'):
        raise ValueError('Require unique existing private checkpoint folder')
    config=configparser.ConfigParser(interpolation=None);config.read(config_path)
    token=json.loads(config['domino-drive']['token'])['access_token']
    validate_sharing(drive_permissions(matches[0]['ID'], token), policy)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('seal', 'unseal', 'preflight'))
    p.add_argument('--state', type=Path)
    p.add_argument('--evidence', type=Path)
    p.add_argument('--source', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--context', type=Path)
    p.add_argument('--expected-segment', type=int, choices=range(FINAL_SEGMENT))
    p.add_argument('--config', type=Path)
    p.add_argument('--verify-drive', action='store_true')
    p.add_argument('--remote', default=PRIVATE_REMOTES[0], choices=PRIVATE_REMOTES)
    p.add_argument('--sharing-policy', default='owner-only', choices=('owner-only','restricted-users'))
    args = p.parse_args()
    key = key_from_environment()
    if args.mode == 'preflight':
        if args.config is not None:
            writable_drive_config(args.config)
        if args.verify_drive:
            if args.config is None:
                raise ValueError('Require explicit private config to verify Drive ACL')
            verify_private_drive(args.config,args.remote,args.sharing_policy)
        print('Dedicated network resume key validated')
        return
    context = json.loads(args.context.read_text())
    if args.mode == 'seal' and args.expected_segment is not None and json.loads(args.state.read_text())['completed_segment'] != args.expected_segment:
        raise ValueError('Wrong completed checkpoint segment')
    result = (seal(args.state, args.evidence, args.output, context, key) if args.mode == 'seal'
              else unseal(args.source, args.output, context, key))
    if args.mode == 'unseal' and args.expected_segment is not None and json.loads((args.output / 'state.json').read_text())['completed_segment'] != args.expected_segment:
        raise ValueError('Wrong restored checkpoint segment')
    print(json.dumps(dict(ciphertext_sha256=result)))


if __name__ == '__main__':
    main()
