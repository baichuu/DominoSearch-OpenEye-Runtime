"""Bounded stage chain, checkpoint retries and private-only state controls."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import yaml
from fpga.openeye.write_clocked_resume_workflows import render,PHASES
from fpga.openeye.clocked_stage_job import choose_checkpoint


def test_all21_stages_have_three_ordered_slots_and_mandatory_final_audit(tmp_path):
    (tmp_path/'.github/workflows').mkdir(parents=True);render(tmp_path)
    data=yaml.safe_load((tmp_path/'.github/workflows/openeye-clocked-resumable.yml').read_text())
    assert set(data['on'])=={'workflow_dispatch'}and len(PHASES)==21
    assert data['permissions']=={'contents':'read'}
    previous=None
    for phase in range(21):
        for slot in range(3):
            name=f'phase_{phase:02d}_{slot}';job=data['jobs'][name]
            assert job['uses']=='./.github/workflows/openeye-clocked-chunk.yml'
            assert job['with']['phase']==phase and job['with']['slot']==slot
            if previous:assert job['needs']==[previous]and f"needs.{previous}.result == 'success'"in job['if']
            if previous and slot==0:assert '.outputs.phase_complete' in job['if']
            if slot:assert '.outputs.phase_complete' in job['with']['skip']
            previous=name
    final=data['jobs']['final_audit'];assert final['needs']==[previous]and 'always()'in final['if']
    assert final['steps'][0]['name']=='Require complete execution of every stage'
    assert any('clocked_stage_job.py final'in s.get('run','')for s in final['steps'])


def test_machine_snapshot_cannot_become_public_artifact(tmp_path):
    (tmp_path/'.github/workflows').mkdir(parents=True);render(tmp_path)
    worker=yaml.safe_load((tmp_path/'.github/workflows/openeye-clocked-chunk.yml').read_text())
    assert set(worker['on'])=={'workflow_call'}and worker['permissions']=={'contents':'read'}
    execute=worker['jobs']['execute'];assert execute['timeout-minutes']==300
    chunk=next(s for s in execute['steps']if s.get('id')=='chunk')
    assert '--wall-seconds 10800'in chunk['run']and '2>&1'in chunk['run']and 'exit "$status"'in chunk['run']
    for s in execute['steps']:
        if s.get('uses','').startswith('actions/upload-artifact@'):
            assert s['with']['path']=='${{ runner.temp }}/public-progress/progress.json'


def test_rerun_failed_slot_prefers_its_own_latest_authenticated_candidate(tmp_path,monkeypatch):
    import fpga.openeye.clocked_stage_job as module
    rows=[dict(Name=n)for n in('phase-04-slot-2-attempt-1.aesgcm','phase-04-slot-2-attempt-3.aesgcm',
        'phase-04-slot-2-attempt-4.aesgcm','phase-03-slot-2-attempt-3.aesgcm','unknown')]
    monkeypatch.setenv('GITHUB_RUN_ATTEMPT','4')
    monkeypatch.setattr(module.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0,stdout=json.dumps(rows)))
    args=SimpleNamespace(checkpoint='phase-04-slot-1-attempt-1.aesgcm',config=tmp_path/'private',phase=4,slot=2)
    assert choose_checkpoint(args,'private')=='phase-04-slot-2-attempt-3.aesgcm'
    args.checkpoint='../outside'
    with pytest.raises(ValueError):choose_checkpoint(args,'private')


@pytest.mark.parametrize('mutation',[None,'manifest','cycles','source','binary','full-pass','order'])
def test_prefix_proof_binds_actual_chunk_source_and_clock(tmp_path,monkeypatch,mutation):
    import fpga.openeye.clocked_stage_job as module
    chain=dict(preparation_sha256='a'*64,binary_sha256='b'*64,chunks=[dict(manifest_sha256='c'*64,
        end_status=dict(observed_cycles=123))])
    report=dict(result='PASS_CHECKED_CLOCKED_PREFIX_NOT_NETWORK',stop_after_instruction=-1,
        preparation_sha256='a'*64,binary_sha256='b'*64,observed_prefix_cycles=123,
        real_rtl_junit=1,full_network_rtl_pass=False,full_network_cycles=None)
    proof=dict(chunk_index=0,manifest_sha256='c'*64,audit=report);checked={'0':proof}
    if mutation=='manifest':proof['manifest_sha256']='d'*64
    elif mutation=='cycles':report['observed_prefix_cycles']=124
    elif mutation=='source':report['preparation_sha256']='d'*64
    elif mutation=='binary':report['binary_sha256']='d'*64
    elif mutation=='full-pass':report['full_network_rtl_pass']=True
    elif mutation=='order':checked={'1':proof}
    (tmp_path/'checked-prefixes.json').write_text(json.dumps(checked))
    (tmp_path/'resume-chain.json').touch()
    monkeypatch.setattr(module,'validate_chain',lambda root:chain)
    monkeypatch.setattr(module,'graph_plan',lambda root:[dict(stop_instruction=-1)])
    if mutation:
        with pytest.raises(ValueError):module.checked_prefixes(tmp_path)
    else:assert module.checked_prefixes(tmp_path)==checked


def test_new_graph_has_no_fabricated_checked_prefix(tmp_path):
    from fpga.openeye.clocked_stage_job import checked_prefixes
    (tmp_path/'checked-prefixes.json').write_text('{}')
    assert checked_prefixes(tmp_path)=={}
    (tmp_path/'checked-prefixes.json').write_text('{"0":{}}')
    with pytest.raises(ValueError):checked_prefixes(tmp_path)


def test_numeric_failure_preserves_private_state_and_never_reexecutes_completed_boundary(tmp_path,monkeypatch):
    import fpga.openeye.clocked_stage_job as module
    import fpga.openeye.audit_clocked_graph as auditor
    root=tmp_path/'graph';root.mkdir();(root/'resume-chain.json').touch()
    release=tmp_path/'release.json';release.write_text('{}');events=[]
    args=SimpleNamespace(profile='optimized',wall_seconds=10800,release=release,workspace=tmp_path,
        phase=1,slot=0,checkpoint='phase-00-slot-0-attempt-1.aesgcm',config=tmp_path/'config',public=tmp_path/'public')
    monkeypatch.setenv('GITHUB_RUN_ATTEMPT','2')
    monkeypatch.setattr(module,'key_from_environment',lambda:b'k'*32)
    monkeypatch.setattr(module,'context',lambda *a:{})
    monkeypatch.setattr(module,'graph_plan',lambda *a:[dict(stop_instruction=-1),dict(stop_instruction=4,name='conv1')])
    monkeypatch.setattr(module,'remote',lambda *a:'private')
    monkeypatch.setattr(module,'choose_checkpoint',lambda *a:args.checkpoint)
    monkeypatch.setattr(module,'fetch',lambda *a:None)
    monkeypatch.setattr(module,'gate_proof',lambda *a:None)
    monkeypatch.setattr(module,'checked_prefixes',lambda *a:{'0':{}})
    monkeypatch.setattr(module,'validate_chain',lambda *a:{'chunks':[dict(index=1,end_status=dict(last_logged_instruction=4))]})
    def no_execution(*a,**k):raise AssertionError('Completed numeric boundary must not execute again')
    monkeypatch.setattr(module,'run',no_execution)
    def fail_audit(*a):raise ValueError('Simulated incorrect actual output')
    monkeypatch.setattr(auditor,'audit',fail_audit)
    def snapshot(*a):events.append('sealed');return 'd'*64
    monkeypatch.setattr(module,'seal',snapshot)
    monkeypatch.setattr(module.subprocess,'run',lambda *a,**k:events.append('uploaded')or SimpleNamespace(returncode=0))
    monkeypatch.setattr(module,'publish_progress',lambda args,phase,done,*rest:events.append(('progress',done)))
    with pytest.raises(ValueError,match='exact private failed state preserved'):module.stage(args)
    assert events==['sealed','uploaded',('progress',False)]
    assert not(root/'checked-prefixes.json').exists()
