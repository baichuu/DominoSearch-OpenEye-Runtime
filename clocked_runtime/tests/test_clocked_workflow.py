"""Public workflow identity, privacy and mandatory gate ordering controls."""
from pathlib import Path
import yaml


def workflow():
    root=Path(__file__).resolve().parents[1]
    path=root/'.github/workflows/openeye-clocked-network.yml'
    if not path.exists():path=root.parent/'.github/workflows/openeye-clocked-network.yml'
    return yaml.safe_load(path.read_text())


def test_three_profiles_use_exact_checkout_and_manual_standard_runner():
    data=workflow();trigger=data.get('on',data.get(True))
    assert set(trigger)=={'workflow_dispatch'}
    assert trigger['workflow_dispatch']['inputs']['profile']['options']==['dense','mac20','optimized']
    assert data['permissions']=={'contents':'read'}
    assert data['env']['OPENEYE_PIN']=='9a742fbd42a55ebfa8fcd39aa41dfa36a367572f'
    job=data['jobs']['whole_network'];assert job['runs-on']=='ubuntu-24.04'and job['timeout-minutes']==360
    assert job['if']=="github.repository == 'baichuu/DominoSearch-OpenEye-Runtime'"
    checkouts=[s['with']for s in job['steps']if s.get('uses','').startswith('actions/checkout@')]
    assert len(checkouts)==2 and all(s['persist-credentials']is False for s in checkouts)
    assert checkouts[0]['ref']=='${{ github.sha }}'
    assert checkouts[1]['repository']=='Learning-Chips-Lab/OpenEye'and checkouts[1]['ref']=='${{ env.OPENEYE_PIN }}'


def test_real_gate_audit_precedes_network_and_only_sealed_artifact_uploads():
    steps=workflow()['jobs']['whole_network']['steps'];names=[s['name']for s in steps]
    gate=names.index('Run actual convolution to FC gate before full network')
    audit=names.index('Audit every raw gate value and forwarded output')
    network=names.index('Execute image to logits on one uninterrupted RTL clock')
    final=names.index('Independently audit graph and export cycle tables')
    assert gate<audit<network<final
    for index in (gate,audit,network,final):
        step=steps[index];assert not step.get('continue-on-error')and not step.get('if')
        assert '> "$RUNNER_TEMP/private-'in step['run']and '2>&1'in step['run']and 'exit "$status"'in step['run']
    uploads=[s for s in steps if s.get('uses','').startswith('actions/upload-artifact@')]
    assert len(uploads)==1 and uploads[0]['with']['path']=='${{ runner.temp }}/public-artifact'
    assert names.index('Seal allowlisted public evidence only')<names.index('Upload counters tables and real JUnit only')
