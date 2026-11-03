"""Fabricated metadata/encryption tests; never real RTL/network evidence."""
import base64
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import yaml

from fpga.openeye.network_checkpoint import (canonical, key_from_environment, public_path,
    seal, unseal, validate_state, validate_prefix_evidence, writable_drive_config, validate_sharing)
from fpga.openeye.run_residual_block import digest
from test_network_evidence import DIGEST, layer_evidence, write


@pytest.fixture
def boundary(tmp_path):
    evidence = tmp_path / 'evidence'
    evidence.mkdir()
    identity = dict(domino_commit='b' * 40, openeye_commit='c' * 40, chain_id='123',
                    checkpoint_sha256=DIGEST, scheme_sha256=DIGEST, packed_manifest_sha256=DIGEST,
                    input_identity=dict(forwarded_int8_sha256=DIGEST))
    a = np.zeros((64,56,56),dtype=np.int64)
    stages = []
    for number in (1,2):
        layer = f'layer1.0.conv{number}'
        name = layer.replace('.','-')
        source = dict(source_layer=layer,source_checkpoint_sha256=DIGEST,source_scheme_sha256=DIGEST,source_packed_manifest_sha256=DIGEST)
        write(evidence / f'{name}-fixture-manifest.json',source)
        stage = dict(layer=layer,backend='openeye-rtl',input_sha256=DIGEST,
                     input_shape=[64,56,56],accumulator_shape=[64,56,56],checked_values=a.size,
                     accumulator_sha256=DIGEST,reference_accumulator_sha256=DIGEST,forwarded_int8_sha256=DIGEST)
        write(evidence / f'{name}-stage-verification.json',stage)
        layer_evidence(evidence/name,64,64,dict(kernel_width=3),a.size,
                       hashlib.sha256((evidence/f'{name}-fixture-manifest.json').read_bytes()).hexdigest())
        stages.append(stage)
    block = dict(block='layer1.0',input_sha256=DIGEST,input_shape=[64,56,56],output_shape=[64,56,56],
                 skip_shape=[64,56,56],output_scale=.125,output_sha256=digest(a),reference_sha256=digest(a),
                 forwarded_int8_sha256=hashlib.sha256(a.astype('i1').tobytes()).hexdigest())
    state = dict(schema_version=2,identity=identity,completed_segment=0,actual=a.tolist(),reference=a.tolist(),
                 scale=.125,stages=stages,blocks=[block])
    path = tmp_path/'state.json'
    write(path,state)
    return state,identity,evidence,path


def test_valid_actual_boundary_and_prefix(boundary):
    state,identity,evidence,_ = boundary
    a,ref,scale = validate_state(state,identity,1)
    assert a.shape==(64,56,56) and np.array_equal(a,ref) and scale==.125
    validate_prefix_evidence(evidence,state)


@pytest.mark.parametrize('change',['identity','order','host','actual','reference','scale','chain','stage-chain','shape','missing-stage','fractional'])
def test_invalid_boundary_rejected(boundary,change):
    state,identity,_,_=boundary
    state=copy.deepcopy(state)
    if change=='identity':state['identity']['chain_id']='999'
    elif change=='order':state['completed_segment']=1
    elif change=='host':state['stages'][0]['backend']='host-reference'
    elif change=='actual':state['actual'][0][0][0]=1
    elif change=='reference':state['reference'][0][0][0]=1
    elif change=='scale':state['scale']=.5
    elif change=='chain':state['blocks'][0]['input_sha256']='wrong'
    elif change=='stage-chain':state['stages'][1]['input_sha256']='wrong'
    elif change=='shape':state['actual'].pop()
    elif change=='missing-stage':state['stages'].pop()
    elif change=='fractional':state['actual'][0][0][0]=.5
    with pytest.raises(ValueError):validate_state(state,identity,1)


def test_real_junit_and_fixture_binding_required(boundary):
    state,_,evidence,_=boundary
    p=evidence/'layer1-0-conv1/spatial-tiles/junit-0000.xml'
    p.write_text('<testsuites><testcase><skipped/></testcase></testsuites>')
    with pytest.raises(ValueError,match='JUnit'):validate_prefix_evidence(evidence,state)
    p.write_text('<testsuites><testcase/></testsuites>')
    write(evidence/'layer1-0-conv1-fixture-manifest.json',dict(source_layer='wrong'))
    with pytest.raises(ValueError,match='source'):validate_prefix_evidence(evidence,state)


@pytest.mark.parametrize('path',['../state.json','/tmp/state.json','weights.json','checkpoint.pt','private-state.json','foo/../tile-0000.json'])
def test_no_private_payload_or_traversal_in_public_checkpoint_evidence(path):
    with pytest.raises(ValueError):public_path(path)


@pytest.mark.parametrize('change',['key','ciphertext','context'])
def test_authenticated_resume_rejects_tamper(boundary,tmp_path,change):
    pytest.importorskip('cryptography')
    state,context,evidence,path=boundary
    key=b'k'*32
    encrypted=tmp_path/'checkpoint.aesgcm'
    seal(path,evidence,encrypted,context,key)
    if change=='key':key=b'x'*32
    elif change=='context':context={**context,'chain_id':'456'}
    else:
        raw=bytearray(encrypted.read_bytes());raw[-1]^=1;encrypted.write_bytes(raw)
    with pytest.raises(ValueError,match='authentication'):unseal(encrypted,tmp_path/'restore',context,key)
    assert not (tmp_path/'restore').exists()


def test_encrypted_actual_boundary_roundtrip_and_exclusive_creation(boundary,tmp_path):
    pytest.importorskip('cryptography')
    state,context,evidence,path=boundary
    encrypted=tmp_path/'checkpoint.aesgcm';key=b'k'*32
    checksum=seal(path,evidence,encrypted,context,key)
    assert encrypted.stat().st_mode & 0o777==0o600
    assert canonical(state) not in encrypted.read_bytes()
    assert unseal(encrypted,tmp_path/'restored',context,key)==checksum
    assert json.loads((tmp_path/'restored/state.json').read_text())==state
    with pytest.raises(ValueError):seal(path,evidence,encrypted,context,key)
    with pytest.raises(ValueError):unseal(encrypted,tmp_path/'restored',context,key)


def test_checkpoint_cli_roundtrip_outside_repository(boundary,tmp_path):
    """Fabricated boundary: exercise the exact direct-script CI import path."""
    pytest.importorskip('cryptography')
    state,context,evidence,path=boundary
    script=Path(__file__).resolve().parents[1]/'fpga/openeye/network_checkpoint.py'
    context_path=tmp_path/'context.json'
    write(context_path,context)
    encrypted=tmp_path/'cli-checkpoint.aesgcm'
    restored=tmp_path/'cli-restored'
    env={k:v for k,v in os.environ.items() if k not in ('PYTHONPATH','DOMINO_RCLONE_CONFIG_B64')}
    env['OPENEYE_NETWORK_RESUME_KEY_B64']=base64.b64encode(b'k'*32).decode()
    # -I prevents pytest's repository imports/PYTHONPATH from hiding CLI failures.
    common=[sys.executable,'-I',str(script)]
    sealed=subprocess.run(common+['seal','--state',str(path),'--evidence',str(evidence),
        '--output',str(encrypted),'--context',str(context_path),'--expected-segment','0'],
        cwd=tmp_path,env=env,text=True,capture_output=True)
    assert sealed.returncode==0,sealed.stderr
    checksum=json.loads(sealed.stdout)['ciphertext_sha256']
    result=subprocess.run(common+['unseal','--source',str(encrypted),'--output',str(restored),
        '--context',str(context_path),'--expected-segment','0'],
        cwd=tmp_path,env=env,text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)['ciphertext_sha256']==checksum
    assert json.loads((restored/'state.json').read_text())==state
    assert encrypted.stat().st_mode & 0o777==0o600
    assert restored.stat().st_mode & 0o777==0o700
    assert env['OPENEYE_NETWORK_RESUME_KEY_B64'] not in sealed.stdout+sealed.stderr+result.stdout+result.stderr


def test_secret_and_write_grant_fail_closed(tmp_path,monkeypatch):
    monkeypatch.delenv('OPENEYE_NETWORK_RESUME_KEY_B64',raising=False)
    with pytest.raises(ValueError):key_from_environment()
    monkeypatch.setenv('OPENEYE_NETWORK_RESUME_KEY_B64',base64.b64encode(b'k'*32).decode())
    assert key_from_environment()==b'k'*32
    monkeypatch.setenv('OPENEYE_NETWORK_RESUME_KEY_B64','invalid-secret')
    with pytest.raises(ValueError) as error:key_from_environment()
    assert 'invalid-secret' not in str(error.value)
    conf=tmp_path/'rclone.conf'
    conf.write_text('[domino-drive]\ntype=drive\nscope=drive.readonly\n')
    with pytest.raises(ValueError,match='write grant'):writable_drive_config(conf)
    conf.write_text('[domino-drive]\ntype=drive\nscope=drive\n')
    writable_drive_config(conf)


@pytest.mark.parametrize('permissions',[
    [dict(type='user',role='owner'),dict(type='anyone',role='reader')],
    [dict(type='user',role='owner'),dict(type='domain',role='reader')],
    [dict(type='user',role='owner'),dict(type='user',role='writer')],
    [],
])
def test_owner_only_checkpoint_policy_rejects_shared_folder(permissions):
    with pytest.raises(ValueError,match='privacy policy'):validate_sharing(permissions,'owner-only')


def test_only_explicit_restricted_user_policy_allows_existing_writer():
    validate_sharing([dict(type='user',role='owner')],'owner-only')
    validate_sharing([dict(type='user',role='owner'),dict(type='user',role='writer')],'restricted-users')
    with pytest.raises(ValueError):
        validate_sharing([dict(type='anyone',role='writer')],'restricted-users')


def test_manual_resumable_job_chain_and_public_upload_only():
    root=Path(__file__).resolve().parents[1]
    workflow=yaml.safe_load((root/'.github/workflows/openeye-full-layer.yml').read_text())
    assert set(workflow.get('on',workflow.get(True)))=={'workflow_dispatch'}
    assert workflow['concurrency']['cancel-in-progress'] is False
    assert "inputs.scope == 'hybrid-network-resumable' && 'network-fc-smoke'" in workflow['concurrency']['group']
    jobs=workflow['jobs']
    assert 'hybrid-network-resumable' in str(jobs['residual']['strategy']['matrix'])
    for i in range(12):
        job=jobs[f'network_segment_{i}']
        assert job['needs']==('residual' if i==0 else f'network_segment_{i-1}')
        assert job['env']['SEGMENT_INDEX']==str(i) and job['timeout-minutes']==360
        assert 'hybrid-network-resumable' in job['if']
        steps=job['steps']
        uploads=[s for s in steps if s.get('uses','').startswith('actions/upload-artifact')]
        assert {s['with']['path'] for s in uploads} == {
            '${{ runner.temp }}/network-public-evidence', '${{ runner.temp }}/network-report'}
        assert any('Reverify complete same-commit gates' in s['name'] for s in steps)
    assert set(jobs) == {"runtime_preflight", "residual", *{f"network_segment_{i}" for i in range(12)}}


@pytest.mark.parametrize('ingress_change',[
    'none','timestamp','bundle-hash','source-commit','checkpoint-hash',
    'candidate-manifest-hash','candidate-count','claim-level','method','schema',
    'missing-field','extra-field','missing-timestamp',
])
def test_segment_execution_restores_actual_boundary_not_source_or_reference_only(tmp_path,monkeypatch,ingress_change):
    """Fake DUT outputs exercise orchestration only; not an RTL pass."""
    pytest.importorskip('cryptography')
    from types import SimpleNamespace
    import fpga.openeye.run_network_inference as runner
    bundle=tmp_path/'bundle';packed=bundle/'mac20/packed-v1';packed.mkdir(parents=True)
    write(packed/'manifest.json',dict(checkpoint=dict(sha256=DIGEST),scheme=dict(sha256=DIGEST)))
    ingress=dict(schema_version=1,method='fabricated-unit-test',
        verified_utc='2026-10-04T15:07:43.134712+00:00',bundle_manifest_sha256=DIGEST,
        source_branch='fabricated-unit-test',source_commit='a'*40,
        candidates=dict(mac20=dict(checkpoint_sha256=DIGEST,packed_manifest_sha256=DIGEST,layers=21)),
        claim_level='transport-and-structure-only')
    original_ingress=copy.deepcopy(ingress)
    identity=dict(domino_commit='b'*40,chain_id='123',openeye_commit=runner.EXPECTED_OPENEYE_COMMIT,
                  checkpoint_sha256=DIGEST,scheme_sha256=DIGEST,packed_manifest_sha256=DIGEST,
                  input_identity=dict(forwarded_int8_sha256=DIGEST))
    monkeypatch.setattr(runner,'verify_bundle',lambda _:ingress)
    monkeypatch.setattr(runner,'source_activation',lambda *a:(np.zeros((64,56,56),dtype=np.int64),.125,identity['input_identity']))
    monkeypatch.setattr(runner,'resume_identity',lambda *a:identity)
    monkeypatch.setattr(runner.subprocess,'check_output',lambda command,**kw:runner.EXPECTED_OPENEYE_COMMIT+'\n' if 'rev-parse' in command else '')
    calls=[]
    class FakeRuntime:
        def __init__(self,packed,manifest,private,evidence,openeye,reference_only):
            assert not reference_only
            self.evidence=evidence;self.stages=[];self.blocks=[]
        def block(self,prefix,a,ref_a,scale,downsample,rtl):
            assert rtl and not downsample and np.array_equal(a,ref_a)
            index=int(prefix[-1]);assert int(a[0,0,0])==index
            calls.append(prefix)
            forwarded=DIGEST if not self.blocks else self.blocks[-1]['forwarded_int8_sha256']
            for number in (1,2):
                name=f'{prefix}.conv{number}';slug=name.replace('.','-')
                stage=dict(layer=name,backend='openeye-rtl',input_sha256=forwarded,input_shape=[64,56,56],
                           accumulator_shape=[64,56,56],checked_values=a.size,accumulator_sha256=DIGEST,
                           reference_accumulator_sha256=DIGEST,forwarded_int8_sha256=DIGEST)
                source=dict(source_layer=name,source_checkpoint_sha256=DIGEST,source_scheme_sha256=DIGEST,
                            source_packed_manifest_sha256=DIGEST)
                write(self.evidence/f'{slug}-fixture-manifest.json',source)
                write(self.evidence/f'{slug}-stage-verification.json',stage)
                layer_evidence(self.evidence/slug,64,64,dict(kernel_width=3),a.size,
                               hashlib.sha256((self.evidence/f'{slug}-fixture-manifest.json').read_bytes()).hexdigest())
                # Make the fabricated layer input agree with this stage record.
                report_path=self.evidence/slug/'spatial_layer_verification.json'
                report=json.loads(report_path.read_text());report['activation_sha256']=forwarded;write(report_path,report)
                self.stages.append(stage);forwarded=DIGEST
            actual=np.full((64,56,56),index+1,dtype=np.int64)
            self.blocks.append(dict(block=prefix,input_sha256=self.stages[-2]['input_sha256'],input_shape=[64,56,56],
                output_shape=[64,56,56],skip_shape=[64,56,56],output_scale=.125,output_sha256=digest(actual),
                reference_sha256=digest(actual),forwarded_int8_sha256=hashlib.sha256(actual.astype('i1').tobytes()).hexdigest()))
            return actual,actual.copy(),.125
    monkeypatch.setattr(runner,'Runtime',FakeRuntime)
    key=b'k'*32
    monkeypatch.setenv('OPENEYE_NETWORK_RESUME_KEY_B64',base64.b64encode(key).decode())
    def args(index,checkpoint=None):
        return SimpleNamespace(bundle_root=bundle,openeye_root=tmp_path/'OpenEye',private_root=tmp_path/f'private-{index}',
            evidence=tmp_path/f'evidence-{index}',mode='full-network',spatial_size=56,sample=0,reference_only=False,
            segment_index=index,chain_id='123',resume_checkpoint=checkpoint,resume_context_only=None)
    result=runner.execute(args(0))
    assert result['result']=='SEGMENT_PASS' and not result['full_network_complete']
    checkpoint=tmp_path/'boundary.aesgcm'
    seal(tmp_path/'private-0/network-state.json',tmp_path/'evidence-0',checkpoint,identity,key)
    if ingress_change=='timestamp':ingress['verified_utc']='2026-10-04T16:20:55.000000+00:00'
    elif ingress_change=='bundle-hash':ingress['bundle_manifest_sha256']='d'*64
    elif ingress_change=='source-commit':ingress['source_commit']='d'*40
    elif ingress_change=='checkpoint-hash':ingress['candidates']['mac20']['checkpoint_sha256']='d'*64
    elif ingress_change=='candidate-manifest-hash':ingress['candidates']['mac20']['packed_manifest_sha256']='d'*64
    elif ingress_change=='candidate-count':ingress['candidates']['mac20']['layers']=20
    elif ingress_change=='claim-level':ingress['claim_level']='wrong'
    elif ingress_change=='method':ingress['method']='wrong'
    elif ingress_change=='schema':ingress['schema_version']=2
    elif ingress_change=='missing-field':del ingress['source_commit']
    elif ingress_change=='extra-field':ingress['unexpected']='wrong'
    elif ingress_change=='missing-timestamp':del ingress['verified_utc']
    if ingress_change not in ('none','timestamp'):
        with pytest.raises(ValueError,match='Checkpoint ingress differs'):
            runner.execute(args(1,checkpoint))
        assert calls==['layer1.0']  # Reject before invoking a new DUT block.
        return
    result=runner.execute(args(1,checkpoint))
    assert calls==['layer1.0','layer1.1'] and result['completed_blocks']==calls
    assert json.loads((tmp_path/'private-1/network-state.json').read_text())['actual'][0][0][0]==2
    assert not (tmp_path/'evidence-1/network-verification.json').exists()
    assert not any('state.json'==p.name for p in (tmp_path/'evidence-1').rglob('*'))
    assert json.loads((tmp_path/'evidence-1/ingress-verification.json').read_text())==original_ingress
    assert result['ingress_verified_utc']==ingress['verified_utc']


def test_latest_same_run_gate_attempt_used_for_rerun_failed_jobs(tmp_path,monkeypatch):
    import fpga.openeye.select_network_gate as gate
    commit='b'*40
    for attempt in (1,3):
        directory=tmp_path/f'openeye-runtime-downsample-block-pilot-{commit}-{attempt}'
        directory.mkdir();(directory/'marker.txt').write_text(str(attempt))
    seen=[]
    monkeypatch.setattr(gate,'verify',lambda root,mode,sha:seen.append(root.name) or dict(result='PASS'))
    result=gate.select(tmp_path,tmp_path/'selected','downsample-block',commit)
    assert result['gate_attempt']==3 and (tmp_path/'selected/marker.txt').read_text()=='3'
    assert len(seen)==1


def test_bad_latest_gate_does_not_fall_back_to_old_green(tmp_path,monkeypatch):
    import fpga.openeye.select_network_gate as gate
    commit='b'*40
    for attempt in (1,2): (tmp_path/f'openeye-runtime-network-fc-smoke-{commit}-{attempt}').mkdir()
    def fail(*args):raise ValueError('newer gate failed')
    monkeypatch.setattr(gate,'verify',fail)
    with pytest.raises(ValueError,match='newer gate failed'):
        gate.select(tmp_path,tmp_path/'selected','fc-smoke',commit)
    assert not (tmp_path/'selected').exists()
