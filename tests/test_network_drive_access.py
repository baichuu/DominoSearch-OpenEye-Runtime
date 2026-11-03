"""Mock Drive ACL/error responses; no real grant or private checkpoint access."""
import io
import json
import random
import time
import urllib.error
from urllib.parse import parse_qs, urlsplit

import pytest

from fpga.openeye import network_checkpoint as checkpoint


REMOTE = 'domino-drive:DominoSearch-runtime-private'
OWNER = dict(type='user', role='owner')
PRIVATE_MARKER = 'fake-private-token-and-provider-message'


def http_error(status, reason=None):
    body = dict(error=dict(message=PRIVATE_MARKER,
        errors=[] if reason is None else [dict(reason=reason, message=PRIVATE_MARKER)]))
    return urllib.error.HTTPError('https://example.invalid/private-folder', status,
        PRIVATE_MARKER, {}, io.BytesIO(json.dumps(body).encode()))


@pytest.fixture
def drive(tmp_path, monkeypatch):
    config = tmp_path/'rclone.conf'
    config.write_text('[domino-drive]\ntype=drive\nscope=drive\ntoken='+
                      json.dumps(dict(access_token=PRIVATE_MARKER))+'\n')
    monkeypatch.setattr(checkpoint.subprocess, 'check_output', lambda *a, **kw:
        json.dumps([dict(Name='DominoSearch-runtime-private', ID='fake-folder-id')]))
    sleeps = []
    monkeypatch.setattr(time, 'sleep', sleeps.append)
    monkeypatch.setattr(random, 'random', lambda: 0.25)
    requests = []

    def respond(sequence):
        pending = iter(sequence)
        def open_request(request, timeout):
            assert timeout == 30
            assert request.get_method() == 'GET'
            assert request.get_header('Authorization') == 'Bearer '+PRIVATE_MARKER
            requests.append(request)
            value = next(pending)
            if isinstance(value, Exception):
                raise value
            return io.StringIO(value if isinstance(value, str) else json.dumps(value))
        monkeypatch.setattr(checkpoint.urllib.request, 'urlopen', open_request)
    return config, requests, sleeps, respond


@pytest.mark.parametrize('reason', ['rateLimitExceeded', 'userRateLimitExceeded'])
def test_transient_403_retries_then_still_checks_owner(drive, reason, capsys):
    config, requests, sleeps, respond = drive
    respond([http_error(403, reason), dict(permissions=[OWNER])])
    before = config.read_bytes()
    checkpoint.verify_private_drive(config, REMOTE, 'owner-only')
    assert len(requests) == 2 and sleeps == [1.25]
    assert config.read_bytes() == before
    output = capsys.readouterr()
    assert reason in output.err and PRIVATE_MARKER not in output.out+output.err
    assert 'private-folder' not in output.out+output.err


@pytest.mark.parametrize('status', [429, 500, 502, 503, 504])
def test_transient_http_status_retries(drive, status):
    config, requests, sleeps, respond = drive
    respond([http_error(status), dict(permissions=[OWNER])])
    checkpoint.verify_private_drive(config, REMOTE, 'owner-only')
    assert len(requests) == 2 and sleeps == [1.25]


@pytest.mark.parametrize('status,reason', [
    (401, 'authError'), (403, 'insufficientPermissions'), (403, 'domainPolicy'),
    (403, 'dailyLimitExceeded'), (403, 'appNotAuthorizedToFile'),
    (403, None), (403, PRIVATE_MARKER), (404, 'notFound'),
])
def test_access_or_unknown_errors_fail_closed_without_retry(drive, status, reason, capsys):
    config, requests, sleeps, respond = drive
    respond([http_error(status, reason)])
    with pytest.raises(ValueError, match='Drive ACL request failed') as error:
        checkpoint.verify_private_drive(config, REMOTE, 'owner-only')
    output = capsys.readouterr()
    assert PRIVATE_MARKER not in str(error.value)+output.out+output.err
    assert len(requests) == 1 and not sleeps


def test_rate_limit_retries_are_bounded_and_never_bypass_acl(drive, capsys):
    config, requests, sleeps, respond = drive
    respond([http_error(403, 'rateLimitExceeded') for _ in range(8)])
    with pytest.raises(ValueError, match='Drive ACL request failed'):
        checkpoint.verify_private_drive(config, REMOTE, 'owner-only')
    assert len(requests) == 8 and sleeps == [1.25, 2.25, 4.25, 8.25, 16.25, 32.25, 32.25]
    output = capsys.readouterr()
    assert PRIVATE_MARKER not in output.out+output.err


def test_longer_rate_limit_window_recovers_but_still_checks_owner(drive):
    config,requests,sleeps,respond=drive
    respond([http_error(403,'rateLimitExceeded') for _ in range(7)]+[dict(permissions=[OWNER])])
    checkpoint.verify_private_drive(config,REMOTE,'owner-only')
    assert len(requests)==8 and sum(sleeps)>60 and max(sleeps)<33


def test_permission_denial_after_old_retry_window_is_still_fatal(drive):
    config,requests,sleeps,respond=drive
    respond([http_error(403,'rateLimitExceeded') for _ in range(5)]+[http_error(403,'insufficientPermissions')])
    with pytest.raises(ValueError,match='insufficientPermissions'):
        checkpoint.verify_private_drive(config,REMOTE,'owner-only')
    assert len(requests)==6 and len(sleeps)==5


def test_extra_writer_after_long_throttle_window_is_rejected(drive):
    config,requests,_,respond=drive
    respond([http_error(403,'rateLimitExceeded') for _ in range(6)]+
            [dict(permissions=[OWNER,dict(type='user',role='writer')])])
    with pytest.raises(ValueError,match='privacy policy'):
        checkpoint.verify_private_drive(config,REMOTE,'owner-only')
    assert len(requests)==7


def test_segment_checks_acl_once_in_authoritative_helper_before_checkpoint_io():
    from pathlib import Path
    import yaml
    root=Path(__file__).resolve().parents[1]
    workflow=yaml.safe_load((root/'.github/workflows/openeye-full-layer.yml').read_text())
    steps=workflow['jobs']['network_segment_0']['steps']
    setup=next(s for s in steps if s['name']=='Configure private artifact access')
    assert '--verify-drive' not in setup['run']
    shell=(root/'fpga/openeye/run_network_segment.sh').read_text()
    assert '--verify-drive' in shell
    assert shell.index('--verify-drive')<shell.index('rclone mkdir')
    assert shell.index('--verify-drive')<shell.index('rclone lsf')
    assert shell.index('--verify-drive')<shell.index('rclone copyto')


def test_mixed_rate_limit_and_access_denial_is_not_retried(drive):
    config, requests, sleeps, respond = drive
    body = json.dumps(dict(error=dict(errors=[dict(reason='rateLimitExceeded'),
        dict(reason='insufficientPermissions')]))).encode()
    respond([urllib.error.HTTPError('https://example.invalid', 403, PRIVATE_MARKER, {}, io.BytesIO(body))])
    with pytest.raises(ValueError, match='insufficientPermissions'):
        checkpoint.verify_private_drive(config, REMOTE, 'owner-only')
    assert len(requests) == 1 and not sleeps


def test_scope_error_info_is_reported_without_raw_response(drive, capsys):
    config, requests, sleeps, respond = drive
    body = json.dumps(dict(error=dict(message=PRIVATE_MARKER,
        details=[dict(reason='ACCESS_TOKEN_SCOPE_INSUFFICIENT', metadata=dict(secret=PRIVATE_MARKER))]))).encode()
    respond([urllib.error.HTTPError('https://example.invalid', 403, PRIVATE_MARKER, {}, io.BytesIO(body))])
    with pytest.raises(ValueError, match='ACCESS_TOKEN_SCOPE_INSUFFICIENT'):
        checkpoint.verify_private_drive(config, REMOTE, 'owner-only')
    output = capsys.readouterr()
    assert PRIVATE_MARKER not in output.out+output.err
    assert len(requests) == 1 and not sleeps


def test_unparseable_403_body_is_unknown_and_not_retried(drive, capsys):
    config, requests, sleeps, respond = drive
    respond([urllib.error.HTTPError('https://example.invalid', 403, PRIVATE_MARKER, {},
                                    io.BytesIO(PRIVATE_MARKER.encode()))])
    with pytest.raises(ValueError, match='unknown'):
        checkpoint.verify_private_drive(config, REMOTE, 'owner-only')
    output = capsys.readouterr()
    assert PRIVATE_MARKER not in output.out+output.err
    assert len(requests) == 1 and not sleeps


def test_success_after_retry_still_rejects_extra_writer(drive):
    config, requests, _, respond = drive
    respond([http_error(403, 'rateLimitExceeded'),
             dict(permissions=[OWNER, dict(type='user', role='writer')])])
    with pytest.raises(ValueError, match='privacy policy'):
        checkpoint.verify_private_drive(config, REMOTE, 'owner-only')
    assert len(requests) == 2


@pytest.mark.parametrize('error', [TimeoutError(), urllib.error.URLError(PRIVATE_MARKER)])
def test_transport_error_retries_without_printing_exception_body(drive, error, capsys):
    config, requests, sleeps, respond = drive
    respond([error, dict(permissions=[OWNER])])
    checkpoint.verify_private_drive(config, REMOTE, 'owner-only')
    assert len(requests) == 2 and sleeps == [1.25]
    output = capsys.readouterr()
    assert PRIVATE_MARKER not in output.out+output.err


def test_writer_on_second_page_cannot_be_ignored(drive):
    config, requests, _, respond = drive
    respond([dict(permissions=[OWNER], nextPageToken='page&two'),
             dict(permissions=[dict(type='user', role='writer')])])
    with pytest.raises(ValueError, match='privacy policy'):
        checkpoint.verify_private_drive(config, REMOTE, 'owner-only')
    query = parse_qs(urlsplit(requests[1].full_url).query)
    assert query['pageToken'] == ['page&two']
    assert 'nextPageToken' in query['fields'][0]


def test_paginated_owner_only_result_passes(drive):
    config, requests, _, respond = drive
    respond([dict(permissions=[], nextPageToken='two'), dict(permissions=[OWNER])])
    checkpoint.verify_private_drive(config, REMOTE, 'owner-only')
    assert len(requests) == 2


@pytest.mark.parametrize('page', [
    {}, dict(permissions='bad'), dict(permissions=[{}]),
    dict(permissions=[OWNER], nextPageToken=7), PRIVATE_MARKER,
])
def test_malformed_page_fails_without_dumping_private_response(drive, page, capsys):
    config, _, _, respond = drive
    respond([page])
    with pytest.raises(ValueError) as error:
        checkpoint.verify_private_drive(config, REMOTE, 'owner-only')
    output = capsys.readouterr()
    assert PRIVATE_MARKER not in str(error.value)+output.out+output.err


def test_repeated_page_token_fails_closed(drive):
    config, _, _, respond = drive
    respond([dict(permissions=[], nextPageToken='same')]*2)
    with pytest.raises(ValueError, match='pagination'):
        checkpoint.verify_private_drive(config, REMOTE, 'owner-only')


def test_pagination_has_a_hard_bound(drive):
    config, requests, _, respond = drive
    respond([dict(permissions=[], nextPageToken=str(i)) for i in range(10)])
    with pytest.raises(ValueError, match='pagination'):
        checkpoint.verify_private_drive(config, REMOTE, 'owner-only')
    assert len(requests) == 10


def test_outside_authorized_remote_rejected_before_drive_call(drive):
    config, requests, _, _ = drive
    with pytest.raises(ValueError, match='authorized roots'):
        checkpoint.verify_private_drive(config, 'domino-drive:other-folder', 'owner-only')
    assert not requests
