"""Private checkpoint authentication and trace/clock ancestry rejection tests."""
from pathlib import Path
import json
import os
import pytest

from fpga.openeye.clocked_snapshot import encrypt,decrypt,safe_path


def test_streamed_checkpoint_authenticates_every_byte_and_identity(tmp_path):
    source=tmp_path/'private';source.write_bytes(os.urandom(2_100_000));key=os.urandom(32)
    context=dict(commit='a'*40,profile='optimized',run_id='123',source_sha256='b'*64)
    encrypted=tmp_path/'encrypted';encrypt(source,encrypted,context,key)
    actual=tmp_path/'actual';decrypt(encrypted,actual,context,key)
    assert actual.read_bytes()==source.read_bytes()
    with pytest.raises(ValueError):encrypt(source,encrypted,context,key)


@pytest.mark.parametrize('mutation',['payload','tag','identity','profile','stem-lanes','function-counters','key','truncated'])
def test_invalid_checkpoint_never_releases_plaintext(tmp_path,mutation):
    source=tmp_path/'private';source.write_bytes(b'private model and SRAM'*100);key=os.urandom(32)
    context=dict(commit='a'*40,profile='mac20',run_id='123',stem_channel_lanes=3,function_metrics=1);encrypted=tmp_path/'encrypted'
    encrypt(source,encrypted,context,key)
    if mutation=='identity':context['commit']='c'*40
    elif mutation=='profile':context['profile']='dense'
    elif mutation=='stem-lanes':context['stem_channel_lanes']=1
    elif mutation=='function-counters':context['function_metrics']=0
    elif mutation=='key':key=os.urandom(32)
    else:
        data=bytearray(encrypted.read_bytes())
        if mutation=='truncated':data=data[:20]
        else:data[-20 if mutation=='payload'else -1]^=1
        encrypted.write_bytes(data)
    output=tmp_path/'must-not-exist'
    with pytest.raises(ValueError):decrypt(encrypted,output,context,key)
    assert not output.exists()


@pytest.mark.parametrize('name',['../model','/tmp/model','generated/../../model','generated/.git/token','evil.py','clocked-build/obj/run.sh','resume-chunks/latest.model'])
def test_snapshot_rejects_unbound_or_executable_extra_paths(name):
    with pytest.raises(ValueError):safe_path(name)
