"""Model admission controls; fabricated metadata is not DUT evidence."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import pytest
from fpga.openeye.network_model_source import (DENSE_PROFILE,DENSE_IDENTITIES,DENSE_BUNDLE_SHA256,
    selected_model,candidate_name,execution_plan,model_ingress,verify_dense_manifest)
from fpga.openeye.prepare_packed_layer_fixture import DEFAULT_PLAN,load_json
from fpga.openeye.verify_network_evidence import verify
from tests.test_network_evidence import network_evidence,seal


def dense_ingress():
    return dict(method='packed-v1-fpga-ingress-integrity',source_commit='dense-baseline-20261107',
        bundle_manifest_sha256=DENSE_BUNDLE_SHA256,candidates=dict(dense=dict(
            checkpoint_sha256=DENSE_IDENTITIES['checkpoint_sha256'],
            packed_manifest_sha256=DENSE_IDENTITIES['packed_manifest_sha256'],layers=21,schemes={'4:4':21})))


def test_dense_plan_preserves_geometry_order_and_original_mixed_plan():
    original=load_json(DEFAULT_PLAN);dense=execution_plan(DENSE_PROFILE)
    assert all(r['expected_scheme']=='4:4' for r in dense['layers'])
    for old,new in zip(original['layers'],dense['layers']):
        assert {k:v for k,v in old.items()if k!='expected_scheme'}=={k:v for k,v in new.items()if k!='expected_scheme'}
    assert execution_plan('mac20')==original==load_json(DEFAULT_PLAN)


@pytest.mark.parametrize('value',['dense','baseline','',None])
def test_unknown_profile_is_not_silently_mac20(monkeypatch,value):
    monkeypatch.setenv('OPENEYE_MODEL_PROFILE','typo')
    with pytest.raises(ValueError):selected_model(value)


def test_workflow_selects_dense_but_default_api_is_legacy_mac20(monkeypatch):
    monkeypatch.delenv('OPENEYE_MODEL_PROFILE',raising=False);assert selected_model()=='mac20'
    monkeypatch.setenv('OPENEYE_MODEL_PROFILE',DENSE_PROFILE)
    assert selected_model()==DENSE_PROFILE and candidate_name(selected_model())=='dense'


@pytest.mark.parametrize('key,value', [('checkpoint_sha256','f'*64),('packed_manifest_sha256','f'*64),
                                       ('layers',20),('schemes',{'3:4':21})])
def test_dense_ingress_rejects_relabeled_or_changed_source(key,value):
    ingress=dense_ingress();assert model_ingress(ingress,DENSE_PROFILE)['layers']==21
    ingress['candidates']['dense'][key]=value
    with pytest.raises(ValueError):model_ingress(ingress,DENSE_PROFILE)


def test_dense_bundle_identity_not_just_candidate_label():
    ingress=dense_ingress();ingress['bundle_manifest_sha256']='f'*64
    with pytest.raises(ValueError):model_ingress(ingress,DENSE_PROFILE)


@pytest.mark.parametrize('mutation',['mask','scheme','checkpoint'])
def test_dense_manifest_rejects_reconstructed_sparse_source(mutation):
    manifest=dict(checkpoint=dict(sha256=DENSE_IDENTITIES['checkpoint_sha256']),
        scheme=dict(sha256=DENSE_IDENTITIES['scheme_sha256']),
        layers=[dict(n=4,m=4,metadata=None)for _ in range(21)])
    verify_dense_manifest(manifest,DENSE_PROFILE)
    if mutation=='mask':manifest['layers'][0]['metadata']={'file':'mask.bin'}
    elif mutation=='scheme':manifest['scheme']['sha256']='f'*64
    else:manifest['checkpoint']['sha256']='f'*64
    with pytest.raises(ValueError):verify_dense_manifest(manifest,DENSE_PROFILE)


def test_mac20_evidence_cannot_be_relabeled_dense(tmp_path):
    root=network_evidence(tmp_path);p=root/'network-verification.json';d=json.loads(p.read_text())
    d['model_profile']=DENSE_PROFILE;p.write_text(json.dumps(d));seal(root)
    with pytest.raises((ValueError,KeyError)):verify(root,'downsample-block','b'*40)


def test_dense_preflight_records_model_without_claiming_rtl():
    from fpga.openeye.network_actions_preflight import plan
    p=plan('hybrid-network-resumable','full-layer-r3','56','a'*40,'b'*40,model_profile=DENSE_PROFILE)
    assert p['model_profile']==DENSE_PROFILE and p['resume_run_id'] is None
    assert p['requires_same_dispatch_gates'] and not p['rtl_executed']
    with pytest.raises(ValueError):
        plan('hybrid-network-resumable','full-layer-r3','56','a'*40,'b'*40,model_profile='typo')


def test_ci_scopes_legacy_fixture_profile_without_changing_dense_dut_selection():
    import yaml
    workflow=yaml.safe_load((Path(__file__).resolve().parents[1]/'.github/workflows/openeye-full-layer.yml').read_text())
    assert workflow['env']['OPENEYE_MODEL_PROFILE']==DENSE_PROFILE
    seen=0
    for job in workflow['jobs'].values():
        for step in job['steps']:
            profile=step.get('env',{}).get('OPENEYE_MODEL_PROFILE')
            if step['name']=='Verify checkpoint negative controls':
                assert profile=='mac20'
                assert 'tests/test_network_model_source.py' in step['run']
                seen+=1
            elif profile is not None:
                assert profile==DENSE_PROFILE
    assert seen==sum(name.startswith('network_segment_')for name in workflow['jobs'])
    assert seen==12
