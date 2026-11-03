import copy
from pathlib import Path

import pytest
import yaml

from fpga.openeye.network_actions_preflight import plan

COMMIT, PIN = 'a' * 40, 'b' * 40
RUN = dict(id=123, head_sha=COMMIT, path='.github/workflows/openeye-full-layer.yml',
           event='workflow_dispatch', status='completed')


def test_fresh_network_requires_own_gates_and_does_not_claim_rtl():
    result = plan('hybrid-network-resumable', 'full-layer-r3', '56', COMMIT, PIN)
    assert result['requires_same_dispatch_gates']
    assert result['resume_run_id'] is None
    assert not result['rtl_executed'] and not result['full_network_complete']


@pytest.mark.parametrize('scope', ['hybrid-network', 'hybrid-network-resumable'])
@pytest.mark.parametrize('architecture,size', [('v6', '56'), ('full-layer-r3', '8'), ('full-layer-r3', '16')])
def test_reject_misleading_network_hardware_or_crop(scope, architecture, size):
    with pytest.raises(ValueError, match='requires architecture'):
        plan(scope, architecture, size, COMMIT, PIN)


@pytest.mark.parametrize('field,value', [('id', 124), ('head_sha', 'c'*40),
    ('path', '.github/workflows/other.yml'), ('event', 'push'), ('status', 'in_progress')])
def test_wrong_resume_run_rejected_before_simulation(field, value):
    metadata = copy.deepcopy(RUN)
    metadata[field] = value
    with pytest.raises(ValueError, match='same execution commit'):
        plan('hybrid-network-resumable', 'full-layer-r3', '56', COMMIT, PIN, '123', metadata)


def test_same_commit_manual_resume_plan_still_requires_gates():
    result = plan('hybrid-network-resumable', 'full-layer-r3', '56', COMMIT, PIN, '123', RUN)
    assert result['requires_same_dispatch_gates'] and not result['full_network_complete']


@pytest.mark.parametrize('remote,policy', [
    ('domino-drive:DominoSearch-artifacts/runtime-private', 'owner-only'),
    ('domino-drive:DominoSearch-runtime-private', 'restricted-users'),
])
def test_resume_rejects_legacy_root_or_weaker_policy(remote, policy):
    with pytest.raises(ValueError, match='approved private root'):
        plan('hybrid-network-resumable', 'full-layer-r3', '56', COMMIT, PIN,
             private_remote=remote, private_policy=policy)


@pytest.mark.parametrize('run_id,metadata', [('123', None), ('0', RUN), ('../123', RUN), ('', RUN)])
def test_missing_or_malformed_resume_identity(run_id, metadata):
    with pytest.raises(ValueError):
        plan('hybrid-network-resumable', 'full-layer-r3', '56', COMMIT, PIN, run_id, metadata)


def test_single_layer_runtime_scope_remains_independent():
    result = plan('downsample-block-pilot', 'v6', '16', COMMIT, PIN)
    assert not result['full_network'] and not result['requires_same_dispatch_gates']
    with pytest.raises(ValueError):
        plan('downsample-block-pilot', 'v6', '16', COMMIT, PIN, '123', RUN)


def test_costly_runtime_gates_wait_for_dispatch_and_private_storage_preflight():
    root = Path(__file__).resolve().parents[1]
    workflow = yaml.safe_load((root / '.github/workflows/openeye-full-layer.yml').read_text())
    jobs = workflow['jobs']
    assert jobs['residual']['needs'] == 'runtime_preflight'
    assert jobs['network_segment_0']['needs'] == 'residual'
    preflight = jobs['runtime_preflight']
    assert preflight['timeout-minutes'] == 10
    steps = preflight['steps']
    assert steps[0]['with']['ref'] == '${{ github.sha }}'
    access = next(s for s in steps if 'network_checkpoint.py preflight' in s.get('run', ''))
    assert access['if'] == "${{ inputs.scope == 'hybrid-network-resumable' }}"
    assert '--verify-drive' in access['run']
    upload = next(s for s in steps if s.get('uses', '').startswith('actions/upload-artifact'))
    assert upload['with']['path'] == '${{ runner.temp }}/network-preflight.json'
    assert 'DominoSearch-runtime-private' in workflow['env']['OPENEYE_NETWORK_PRIVATE_REMOTE']
