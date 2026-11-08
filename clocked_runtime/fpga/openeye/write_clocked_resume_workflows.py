#!/usr/bin/env python3
"""Render manual21-stage snapshot workflow; three bounded slots per stage."""
from copy import deepcopy
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[2]
REPOSITORY="github.repository == 'baichuu/DominoSearch-OpenEye-Runtime'"
PHASES=['stem']
for group in range(1,5):
    for block in range(2):
        PHASES.extend([f'layer{group}.{block}.conv1',f'layer{group}.{block}.conv2'])
        if group>1 and block==0:PHASES.append(f'layer{group}.{block}.projection')
PHASES.append('GAP and FC')


def setup():
    initial='inputs.phase == 0 && inputs.slot == 0'
    return [
      dict(name='Check out exact public execution source',uses='actions/checkout@v4',with_={'ref':'${{ github.sha }}','path':'Runtime','persist-credentials':False}),
      dict(name='Check out pinned source for initial controls',uses='actions/checkout@v4',if_=initial,with_={'repository':'Learning-Chips-Lab/OpenEye','ref':'${{ env.OPENEYE_PIN }}','path':'OpenEye','persist-credentials':False}),
      dict(name='Set up Python',uses='actions/setup-python@v5',with_={'python-version':'3.11'}),
      dict(name='Install bounded stage dependencies',run='sudo apt-get update\nsudo apt-get install -y iverilog make patch rclone\npython -m venv OpenEye/.venv\nOpenEye/.venv/bin/python -m pip install -r Runtime/clocked_runtime/fpga/openeye/requirements-clocked.txt -r Runtime/clocked_runtime/fpga/openeye/requirements-network-resume.txt\n'),
      dict(name='Set up exact simulator for initial model build',uses='veryl-lang/setup-verilator@b155747595eb4b817473a83045ba1da581c4084b',if_=initial,with_={'version':'5.052'}),
      dict(name='Initial rejection controls and pinned PE',if_=initial,working_directory='Runtime/clocked_runtime',env={'OPENEYE_NEW_PIN_ROOT':'${{ github.workspace }}/OpenEye'},run='test "$(git -C ../../OpenEye rev-parse HEAD)" = "$OPENEYE_PIN"\ntest -z "$(git -C ../../OpenEye status --short)"\n../../OpenEye/.venv/bin/python -m pytest -q tests/test_clocked_graph_validation.py tests/test_clocked_release_ingress.py tests/test_clocked_snapshot.py tests/test_clocked_resume_control.py tests/test_clocked_resume_workflow.py tests/test_clocked_workflow.py\nfpga/openeye/run_pe_smoke.sh ../../OpenEye "$RUNNER_TEMP/pinned-pe-smoke.log"\n'),
      dict(name='Verify original immutable inputs and owner-only access',working_directory='Runtime/clocked_runtime',env={'DOMINO_RCLONE_CONFIG_B64':'${{ secrets.DOMINO_RCLONE_CONFIG_B64 }}','CLOCKED_PROFILE':'${{ inputs.profile }}'},run='python fpga/openeye/prepare_ci_rclone_config.py --output "$RUNNER_TEMP/clocked-rclone.conf"\n../../OpenEye/.venv/bin/python fpga/openeye/clocked_release_ingress.py --release fpga/openeye/clocked-release.json --profile "$CLOCKED_PROFILE" --config "$RUNNER_TEMP/clocked-rclone.conf" --output "$RUNNER_TEMP/clocked-inputs" > "$RUNNER_TEMP/private-ingress.log" 2>&1\n')]


def normalize(value):
    if isinstance(value,list):return[normalize(v)for v in value]
    if isinstance(value,dict):return {dict(with_='with',if_='if',working_directory='working-directory').get(k,k):normalize(v)for k,v in value.items()}
    return value


def render(root=ROOT):
    workflow=dict(name='OpenEye clocked snapshot stages',**{'run-name':'Resumable ${{ inputs.profile }} ·21RTL stages · pin9a','on':{'workflow_dispatch':{'inputs':{'profile':{'description':'Matched image-to-logit save-restore experiment','required':True,'default':'dense','type':'choice','options':['dense','mac20','optimized']}}}},'permissions':{'contents':'read'},'concurrency':{'group':'openeye-clocked-resumable-${{ github.ref }}-${{ inputs.profile }}','cancel-in-progress':False},'jobs':{}})
    previous=None
    for phase,name in enumerate(PHASES):
        for slot in range(3):
            job_id=f'phase_{phase:02d}_{slot}';job={'name':f'{phase:02d} {name} · slot{slot+1}','uses':'./.github/workflows/openeye-clocked-chunk.yml',
                'secrets':{'DOMINO_RCLONE_CONFIG_B64':'${{ secrets.DOMINO_RCLONE_CONFIG_B64 }}','OPENEYE_NETWORK_RESUME_KEY_B64':'${{ secrets.OPENEYE_NETWORK_RESUME_KEY_B64 }}'},
                'with':{'profile':'${{ inputs.profile }}','phase':phase,'slot':slot,'skip':False,'checkpoint':''}}
            if previous:
                job['needs']=[previous]
                condition=f"always() && {REPOSITORY} && needs.{previous}.result == 'success'"
                if slot==0:condition+=f" && needs.{previous}.outputs.phase_complete == 'true'"
                job['if']=condition
                job['with']['checkpoint']='${{ needs.'+previous+'.outputs.checkpoint_name }}'
                if slot:job['with']['skip']='${{ needs.'+previous+".outputs.phase_complete == 'true' }}"
            else:job['if']=REPOSITORY
            workflow['jobs'][job_id]=job;previous=job_id
    final_steps=setup()
    final_steps=[step for step in final_steps if step.get('if_')is None]
    final_steps.insert(0,dict(name='Require complete execution of every stage',env={'FINAL_COMPLETE':'${{ needs.'+previous+'.outputs.phase_complete }}'},run='test "$FINAL_COMPLETE" = "true" || { echo "Incomplete stage chain; Re-run failed jobs continues its authenticated snapshot."; exit 1; }'))
    final_steps.append(dict(name='Independently audit complete resumed RTL graph',working_directory='Runtime/clocked_runtime',env={'OPENEYE_NETWORK_RESUME_KEY_B64':'${{ secrets.OPENEYE_NETWORK_RESUME_KEY_B64 }}','CLOCKED_PROFILE':'${{ inputs.profile }}','CLOCKED_CHECKPOINT':'${{ needs.'+previous+'.outputs.checkpoint_name }}'},run='set +e\n../../OpenEye/.venv/bin/python fpga/openeye/clocked_stage_job.py final --release fpga/openeye/clocked-release.json --workspace "$RUNNER_TEMP/clocked-inputs" --config "$RUNNER_TEMP/clocked-rclone.conf" --public "$RUNNER_TEMP/public-evidence" --profile "$CLOCKED_PROFILE" --checkpoint "$CLOCKED_CHECKPOINT" > "$RUNNER_TEMP/private-final-audit.log" 2>&1\nstatus=$?\nset -e\nprintf "Independent resumed full-network audit exit status: %s\\n" "$status"\nexit "$status"\n'))
    final_steps.extend([
      dict(name='Seal allowlisted public aggregate evidence',if_='always()',working_directory='Runtime/clocked_runtime',env={'CLOCKED_PROFILE':'${{ inputs.profile }}'},run='python fpga/openeye/seal_clocked_public_evidence.py --root "$RUNNER_TEMP/public-evidence" --output "$RUNNER_TEMP/public-artifact" --profile "$CLOCKED_PROFILE" --execution-commit "$GITHUB_SHA" --run-id "$GITHUB_RUN_ID"'),
      dict(name='Publish audited counters and real JUnit only',if_='always()',uses='actions/upload-artifact@v4',with_={'name':'clocked-resumed-final-${{ inputs.profile }}-${{ github.run_id }}-${{ github.run_attempt }}','path':'${{ runner.temp }}/public-artifact','if-no-files-found':'error','retention-days':30})])
    workflow['jobs']['final_audit']={'name':'Independent complete-network audit','if':'always() && '+REPOSITORY,'needs':[previous],'runs-on':'ubuntu-24.04','timeout-minutes':300,'env':{'OPENEYE_PIN':'9a742fbd42a55ebfa8fcd39aa41dfa36a367572f'},'steps':final_steps}
    outputs={'phase_complete':{'value':'${{ jobs.execute.outputs.phase_complete || jobs.noop.outputs.phase_complete }}'},'checkpoint_name':{'value':'${{ jobs.execute.outputs.checkpoint_name || jobs.noop.outputs.checkpoint_name }}'}}
    inputs={'profile':{'type':'string','required':True},'phase':{'type':'number','required':True},'slot':{'type':'number','required':True},'checkpoint':{'type':'string','default':''},'skip':{'type':'boolean','default':False}}
    worker={'name':'One bounded clocked snapshot stage','on':{'workflow_call':{'inputs':inputs,'secrets':{'DOMINO_RCLONE_CONFIG_B64':{'required':True},'OPENEYE_NETWORK_RESUME_KEY_B64':{'required':True}},'outputs':outputs}},'permissions':{'contents':'read'},'env':{'OPENEYE_PIN':'9a742fbd42a55ebfa8fcd39aa41dfa36a367572f'},'jobs':{}}
    worker['jobs']['noop']={'if':REPOSITORY+' && inputs.skip','runs-on':'ubuntu-24.04','timeout-minutes':5,'outputs':{'phase_complete':'${{ steps.propagate.outputs.phase_complete }}','checkpoint_name':'${{ steps.propagate.outputs.checkpoint_name }}'},'steps':[dict(name='Reuse completed stage without executing RTL',id='propagate',env={'PREVIOUS_CHECKPOINT':'${{ inputs.checkpoint }}'},run='printf "phase_complete=true\\ncheckpoint_name=%s\\n" "$PREVIOUS_CHECKPOINT" >> "$GITHUB_OUTPUT"')]}
    steps=setup()
    steps.append(dict(name='Execute or restore exact machine state',id='chunk',working_directory='Runtime/clocked_runtime',env={'OPENEYE_NETWORK_RESUME_KEY_B64':'${{ secrets.OPENEYE_NETWORK_RESUME_KEY_B64 }}','CLOCKED_PROFILE':'${{ inputs.profile }}','CLOCKED_CHECKPOINT':'${{ inputs.checkpoint }}','CLOCKED_PHASE':'${{ inputs.phase }}','CLOCKED_SLOT':'${{ inputs.slot }}'},run='set +e\n../../OpenEye/.venv/bin/python fpga/openeye/clocked_stage_job.py stage --release fpga/openeye/clocked-release.json --workspace "$RUNNER_TEMP/clocked-inputs" --config "$RUNNER_TEMP/clocked-rclone.conf" --public "$RUNNER_TEMP/public-progress" --profile "$CLOCKED_PROFILE" --phase "$CLOCKED_PHASE" --slot "$CLOCKED_SLOT" --checkpoint "$CLOCKED_CHECKPOINT" --wall-seconds 10800 > "$RUNNER_TEMP/private-snapshot-stage.log" 2>&1\nstatus=$?\nset -e\nprintf "Snapshot stage exit status: %s\\n" "$status"\nexit "$status"\n'))
    steps.append(dict(name='Publish allowlisted progress only',if_='always()',uses='actions/upload-artifact@v4',with_={'name':'clocked-progress-${{ inputs.profile }}-${{ inputs.phase }}-${{ inputs.slot }}-${{ github.run_attempt }}','path':'${{ runner.temp }}/public-progress/progress.json','if-no-files-found':'ignore','retention-days':30}))
    worker['jobs']['execute']={'if':REPOSITORY+' && !inputs.skip','runs-on':'ubuntu-24.04','timeout-minutes':300,'outputs':{'phase_complete':'${{ steps.chunk.outputs.phase_complete }}','checkpoint_name':'${{ steps.chunk.outputs.checkpoint_name }}'},'steps':steps}
    for name,data in [('openeye-clocked-resumable.yml',workflow),('openeye-clocked-chunk.yml',worker)]:
        (root/'.github/workflows'/name).write_text(yaml.safe_dump(normalize(data),sort_keys=False,width=120,allow_unicode=True))


if __name__=='__main__':render()
