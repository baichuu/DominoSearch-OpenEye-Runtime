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
            assert job['with']['stem_channel_lanes']=='${{ inputs.stem_channel_lanes }}'
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
    monkeypatch.setattr(module,'verify_stem_policy',lambda *a:None)
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


def test_requested_stem_hardware_reaches_every_job_and_final_audit(tmp_path):
    (tmp_path/'.github/workflows').mkdir(parents=True);render(tmp_path)
    main=yaml.safe_load((tmp_path/'.github/workflows/openeye-clocked-resumable.yml').read_text())
    choice=main['on']['workflow_dispatch']['inputs']['stem_channel_lanes']
    assert choice['default']=='1'and choice['options']==['1','3']
    assert 'inputs.stem_channel_lanes'in main['concurrency']['group']
    worker=yaml.safe_load((tmp_path/'.github/workflows/openeye-clocked-chunk.yml').read_text())
    assert worker['on']['workflow_call']['inputs']['stem_channel_lanes']['default']=='1'
    chunk=next(v for v in worker['jobs']['execute']['steps']if v.get('id')=='chunk')
    final=next(v for v in main['jobs']['final_audit']['steps']if 'clocked_stage_job.py final'in v.get('run',''))
    for step in(chunk,final):
        assert step['env']['CLOCKED_STEM_CHANNEL_LANES']=='${{ inputs.stem_channel_lanes }}'
        assert '--stem-channel-lanes "$CLOCKED_STEM_CHANNEL_LANES"'in step['run']


@pytest.mark.parametrize('requested,actual,valid',[(1,1,True),(3,3,True),(3,1,False),(1,3,False),(2,1,False),(True,1,False)])
def test_scalar_and_parallel_hardware_cannot_be_mislabeled_or_restored(tmp_path,monkeypatch,requested,actual,valid):
    import fpga.openeye.clocked_resume as resume
    from fpga.openeye.clocked_stage_job import verify_stem_policy
    monkeypatch.setattr(resume,'identity',lambda root:dict(parameters=dict(STEM_CHANNEL_LANES=actual)))
    args=SimpleNamespace(stem_channel_lanes=requested)
    if valid:verify_stem_policy(args,tmp_path)
    else:
        with pytest.raises(ValueError):verify_stem_policy(args,tmp_path)


def test_authenticated_context_binds_requested_stem_lane_count(monkeypatch):
    import fpga.openeye.clocked_stage_job as module
    monkeypatch.setenv('GITHUB_SHA','a'*40);monkeypatch.setenv('GITHUB_RUN_ID','123')
    monkeypatch.setenv('GITHUB_REPOSITORY','baichuu/DominoSearch-OpenEye-Runtime')
    monkeypatch.setattr(module.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0,stdout='a'*40))
    release=dict(profiles=dict(optimized=dict(archive_sha256='b'*64,graph_preparation_sha256='c'*64)),arithmetic_contract_sha256='d'*64)
    scalar=module.context(SimpleNamespace(profile='optimized',stem_channel_lanes=1),release)
    parallel=module.context(SimpleNamespace(profile='optimized',stem_channel_lanes=3),release)
    assert scalar['stem_channel_lanes']==1 and parallel['stem_channel_lanes']==3
    assert {k:v for k,v in scalar.items()if k!='stem_channel_lanes'}=={k:v for k,v in parallel.items()if k!='stem_channel_lanes'}


def test_initial_gate_stays_native_and_graph_build_receives_requested_stem_lanes(tmp_path,monkeypatch):
    import fpga.openeye.clocked_stage_job as module
    import fpga.openeye.run_graph_program as runner
    import fpga.openeye.audit_graph_transition_gate as gate_audit
    (tmp_path/'gate').mkdir();(tmp_path/'graph').mkdir();builds=[]
    def fake_build(args):builds.append((args.root.name,getattr(args,'stem_channel_lanes',1)))
    def fake_run(root,**kwargs):(root/'execution.json').write_text('{"complete":true}')
    monkeypatch.setattr(runner,'execute',fake_build)
    monkeypatch.setattr(module,'run',fake_run)
    monkeypatch.setattr(module,'validate_chain',lambda *a,**k:None)
    monkeypatch.setattr(gate_audit,'audit',lambda *a:dict(result='FABRICATED_ORCHESTRATION_CONTROL_NOT_DUT_EVIDENCE'))
    module.prepare_initial(SimpleNamespace(workspace=tmp_path,stem_channel_lanes=3))
    assert builds==[('gate',1),('graph',3)]


def test_function_counter_policy_reaches_every_slot_and_final_audit(tmp_path):
    (tmp_path/'.github/workflows').mkdir(parents=True);render(tmp_path)
    main=yaml.safe_load((tmp_path/'.github/workflows/openeye-clocked-resumable.yml').read_text())
    choice=main['on']['workflow_dispatch']['inputs']['function_metrics']
    assert choice['default']=='1'and choice['options']==['1','0']
    for job in main['jobs'].values():
        if 'uses'in job:assert job['with']['function_metrics']=='${{ inputs.function_metrics }}'
    worker=yaml.safe_load((tmp_path/'.github/workflows/openeye-clocked-chunk.yml').read_text())
    chunk=next(v for v in worker['jobs']['execute']['steps']if v.get('id')=='chunk')
    final=next(v for v in main['jobs']['final_audit']['steps']if 'clocked_stage_job.py final'in v.get('run',''))
    for step in(chunk,final):
        assert step['env']['CLOCKED_FUNCTION_METRICS']=='${{ inputs.function_metrics }}'
        assert '--function-metrics "$CLOCKED_FUNCTION_METRICS"'in step['run']


@pytest.mark.parametrize('requested,actual,valid',[(0,0,True),(1,1,True),(1,0,False),(0,1,False),(2,0,False),(True,1,False)])
def test_model_counter_policy_cannot_be_mislabeled(tmp_path,monkeypatch,requested,actual,valid):
    import fpga.openeye.clocked_resume as resume
    from fpga.openeye.clocked_stage_job import verify_stem_policy
    monkeypatch.setattr(resume,'identity',lambda root:dict(parameters=dict(STEM_CHANNEL_LANES=1,FUNCTION_METRICS=actual)))
    args=SimpleNamespace(stem_channel_lanes=1,function_metrics=requested)
    if valid:verify_stem_policy(args,tmp_path)
    else:
        with pytest.raises(ValueError):verify_stem_policy(args,tmp_path)


def test_counter_policy_is_authenticated_with_execution_identity(monkeypatch):
    import fpga.openeye.clocked_stage_job as module
    monkeypatch.setenv('GITHUB_SHA','a'*40);monkeypatch.setenv('GITHUB_RUN_ID','123')
    monkeypatch.setenv('GITHUB_REPOSITORY','baichuu/DominoSearch-OpenEye-Runtime')
    monkeypatch.setattr(module.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0,stdout='a'*40))
    release=dict(profiles=dict(optimized=dict(archive_sha256='b'*64,graph_preparation_sha256='c'*64)),arithmetic_contract_sha256='d'*64)
    off=module.context(SimpleNamespace(profile='optimized',function_metrics=0),release)
    on=module.context(SimpleNamespace(profile='optimized',function_metrics=1),release)
    assert off['function_metrics']==0 and on['function_metrics']==1
    assert {k:v for k,v in off.items()if k!='function_metrics'}=={k:v for k,v in on.items()if k!='function_metrics'}


def test_postops_policy_reaches_every_slot_and_final_audit(tmp_path):
    (tmp_path/'.github/workflows').mkdir(parents=True);render(tmp_path)
    main=yaml.safe_load((tmp_path/'.github/workflows/openeye-clocked-resumable.yml').read_text())
    assert 'inputs.postops_pipelined'in main['run-name']
    choice=main['on']['workflow_dispatch']['inputs']['postops_pipelined']
    assert choice['default']=='0'and choice['options']==['0','1']
    for job in main['jobs'].values():
        if 'uses'in job:assert job['with']['postops_pipelined']=='${{ inputs.postops_pipelined }}'
    worker=yaml.safe_load((tmp_path/'.github/workflows/openeye-clocked-chunk.yml').read_text())
    chunk=next(v for v in worker['jobs']['execute']['steps']if v.get('id')=='chunk')
    final=next(v for v in main['jobs']['final_audit']['steps']if 'clocked_stage_job.py final'in v.get('run',''))
    for step in(chunk,final):
        assert step['env']['CLOCKED_POSTOPS_PIPELINED']=='${{ inputs.postops_pipelined }}'
        assert '--postops-pipelined "$CLOCKED_POSTOPS_PIPELINED"'in step['run']


@pytest.mark.parametrize('requested,actual,valid',[(0,0,True),(1,1,True),(1,0,False),(0,1,False),(2,0,False),(True,1,False)])
def test_model_postops_policy_cannot_be_mislabeled(tmp_path,monkeypatch,requested,actual,valid):
    import fpga.openeye.clocked_resume as resume
    from fpga.openeye.clocked_stage_job import verify_stem_policy
    monkeypatch.setattr(resume,'identity',lambda root:dict(parameters=dict(STEM_CHANNEL_LANES=1,POSTOPS_PIPELINED=actual)))
    args=SimpleNamespace(stem_channel_lanes=1,postops_pipelined=requested)
    if valid:verify_stem_policy(args,tmp_path)
    else:
        with pytest.raises(ValueError):verify_stem_policy(args,tmp_path)


def test_initial_native_gate_and_graph_use_same_requested_postops_policy(tmp_path,monkeypatch):
    import fpga.openeye.clocked_stage_job as module
    import fpga.openeye.run_graph_program as runner
    import fpga.openeye.audit_graph_transition_gate as gate_audit
    (tmp_path/'gate').mkdir();(tmp_path/'graph').mkdir();builds=[]
    monkeypatch.setattr(runner,'execute',lambda args:builds.append((args.root.name,getattr(args,'postops_pipelined',False))))
    monkeypatch.setattr(module,'run',lambda root,**kwargs:(root/'execution.json').write_text('{"complete":true}'))
    monkeypatch.setattr(module,'validate_chain',lambda *a,**k:None)
    monkeypatch.setattr(gate_audit,'audit',lambda *a:dict(result='FABRICATED_ORCHESTRATION_CONTROL_NOT_DUT_EVIDENCE'))
    module.prepare_initial(SimpleNamespace(workspace=tmp_path,postops_pipelined=1))
    assert builds==[('gate',True),('graph',True)]


def test_postops_controller_policy_is_authenticated(monkeypatch):
    import fpga.openeye.clocked_stage_job as module
    monkeypatch.setenv('GITHUB_SHA','a'*40);monkeypatch.setenv('GITHUB_RUN_ID','123')
    monkeypatch.setenv('GITHUB_REPOSITORY','baichuu/DominoSearch-OpenEye-Runtime')
    monkeypatch.setattr(module.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0,stdout='a'*40))
    release=dict(profiles=dict(optimized=dict(archive_sha256='b'*64,graph_preparation_sha256='c'*64)),arithmetic_contract_sha256='d'*64)
    old=module.context(SimpleNamespace(profile='optimized',postops_pipelined=0),release)
    new=module.context(SimpleNamespace(profile='optimized',postops_pipelined=1),release)
    assert old['postops_pipelined']==0 and new['postops_pipelined']==1
    assert {k:v for k,v in old.items()if k!='postops_pipelined'}=={k:v for k,v in new.items()if k!='postops_pipelined'}


@pytest.mark.parametrize('actual,requested',[(False,1),(True,0)])
def test_native_gate_policy_mismatch_blocks_full_graph(tmp_path,actual,requested):
    from fpga.openeye.clocked_stage_job import gate_proof
    report=dict(result='PASS_ACTUAL_CONV_TO_NATIVE_FC_PROTOCOL_GATE',openeye_pin='9a742fbd42a55ebfa8fcd39aa41dfa36a367572f',
        preparation_sha256='a'*64,real_rtl_junit=1,postops_pipelined=actual)
    (tmp_path/'gate-audit.json').write_text(json.dumps(report))
    (tmp_path/'gate-junit.xml').write_text('<testsuite tests="1" failures="0"><testcase/></testsuite>')
    release=dict(profiles=dict(optimized=dict(gate_preparation_sha256='a'*64)))
    with pytest.raises(ValueError,match='policy differs'):gate_proof(tmp_path,release,'optimized',requested)
