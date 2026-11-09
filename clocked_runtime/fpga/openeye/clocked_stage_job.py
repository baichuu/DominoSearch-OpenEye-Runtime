#!/usr/bin/env python3
"""One bounded Actions stage; restore the exact RTL state, never host outputs."""
import argparse
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from fpga.openeye.clocked_release_ingress import verify_release_folder,PRIVATE_ROOT
from fpga.openeye.clocked_resume import graph_plan,run,validate_chain,sha
from fpga.openeye.clocked_snapshot import seal,restore
from fpga.openeye.network_checkpoint import key_from_environment,verify_private_drive
from fpga.openeye.validate_graph_program import require,PIN

NAME=r'phase-\d{2}-slot-[012]-attempt-\d+\.aesgcm'


def stem_lanes(args):
    value=getattr(args,'stem_channel_lanes',1)
    require(type(value)is int and value in(1,3),'Unsupported stem channel lane policy')
    return value


def function_metrics(args):
    value=getattr(args,'function_metrics',0)
    require(type(value)is int and value in(0,1),'Unsupported function counter policy')
    return value


def postops_pipeline(args):
    value=getattr(args,'postops_pipelined',0)
    require(type(value)is int and value in(0,1),'Unsupported postops controller policy')
    return value


def verify_stem_policy(args,root):
    from fpga.openeye.clocked_resume import identity
    parameters=identity(root)['parameters']
    require(parameters.get('STEM_CHANNEL_LANES',1)==stem_lanes(args),
            'Actual model stem lanes differ from requested checkpoint policy')
    require(parameters.get('FUNCTION_METRICS',0)==function_metrics(args),
            'Actual model function counters differ from requested checkpoint policy')
    require(parameters.get('POSTOPS_PIPELINED',0)==postops_pipeline(args),
            'Actual model postops controller differs from requested checkpoint policy')


def context(args,release):
    commit=os.environ.get('GITHUB_SHA','');run_id=os.environ.get('GITHUB_RUN_ID','')
    require(re.fullmatch('[0-9a-f]{40}',commit)and re.fullmatch('[0-9]+',run_id),'Exact current public execution identity required')
    require(os.environ.get('GITHUB_REPOSITORY')=='baichuu/DominoSearch-OpenEye-Runtime','Wrong execution repository')
    checkout=subprocess.run(['git','-C',str(ROOT),'rev-parse','HEAD'],capture_output=True,text=True)
    require(checkout.returncode==0 and checkout.stdout.strip()==commit,'Checkout differs from snapshot execution commit')
    return dict(protocol='domino-clocked-snapshot-v1',experiment='whole-network-int8-fixedpoint-v1',
        execution_commit=commit,run_id=run_id,profile=args.profile,openeye_pin=PIN,
        stem_channel_lanes=stem_lanes(args),
        function_metrics=function_metrics(args),
        postops_pipelined=postops_pipeline(args),
        input_archive_sha256=release['profiles'][args.profile]['archive_sha256'],
        preparation_sha256=release['profiles'][args.profile]['graph_preparation_sha256'],
        arithmetic_contract_sha256=release['arithmetic_contract_sha256'])


def remote(args,ctx):
    folder='clocked-state-'+ctx['run_id']+'-'+args.profile+'-'+ctx['execution_commit'][:12]+'-s'+str(stem_lanes(args))+'-m'+str(function_metrics(args))+'-p'+str(postops_pipeline(args))
    verify_private_drive(args.config,PRIVATE_ROOT,'owner-only')
    result=subprocess.run(['rclone','mkdir','--config',str(args.config),PRIVATE_ROOT+'/'+folder],capture_output=True,text=True)
    require(result.returncode==0,'Cannot create existing-grant private checkpoint directory')
    verify_release_folder(args.config,folder);return PRIVATE_ROOT+'/'+folder


def choose_checkpoint(args,remote_name):
    selected=args.checkpoint or None
    if selected:require(re.fullmatch(NAME,selected)is not None,'Unsafe checkpoint identifier')
    result=subprocess.run(['rclone','lsjson','--files-only','--config',str(args.config),remote_name],capture_output=True,text=True)
    require(result.returncode==0,'Cannot list private checkpoints')
    attempt=int(os.environ['GITHUB_RUN_ATTEMPT']);prefix=f'phase-{args.phase:02d}-slot-{args.slot}-attempt-'
    candidates=[]
    for row in json.loads(result.stdout):
        name=row['Name']
        if name.startswith(prefix)and re.fullmatch(NAME,name):
            previous=int(name[len(prefix):-len('.aesgcm')])
            if previous<attempt:candidates.append((previous,name))
    if candidates:selected=max(candidates)[1]
    return selected


def fetch(name,remote_name,args,ctx):
    require(name is not None,'Missing authenticated state from previous completed job')
    path=args.workspace.parent/'restored-clocked-state.aesgcm'
    result=subprocess.run(['rclone','copyto','--immutable','--config',str(args.config),remote_name+'/'+name,str(path)],capture_output=True,text=True)
    require(result.returncode==0,'Private checkpoint retrieval failed')
    restore(path,args.workspace/'graph',ctx,key_from_environment());return path


def gate_proof(root,release,profile,postops_pipelined=0):
    import xml.etree.ElementTree as ET
    report=json.loads((root/'gate-audit.json').read_text());suite=ET.parse(root/'gate-junit.xml').getroot()
    require(report['result']=='PASS_ACTUAL_CONV_TO_NATIVE_FC_PROTOCOL_GATE'and report['openeye_pin']==PIN and
            report['preparation_sha256']==release['profiles'][profile]['gate_preparation_sha256']and report['real_rtl_junit']==1,
            'Wrong or missing same-execution actual gate')
    require(report.get('postops_pipelined',False)==bool(postops_pipelined),'Native gate postops policy differs from full graph')
    require(suite.get('tests')=='1'and all(suite.get(k,'0')=='0'for k in('failures','errors','skipped'))and
            len(list(suite.iter('testcase')))==1 and not any(list(suite.iter(k))for k in('failure','error','skipped')),'Actual native gate JUnit is not PASS')


def prepare_initial(args):
    from fpga.openeye.run_graph_program import execute
    from fpga.openeye.audit_graph_transition_gate import audit
    import xml.etree.ElementTree as ET
    gate=args.workspace/'gate';graph=args.workspace/'graph'
    execute(SimpleNamespace(root=gate,compiled_root=None,compile_only=True,savable=True,
                            postops_pipelined=bool(postops_pipeline(args))))
    run(gate,stop_edges=300000,wall_seconds=10800)
    if not json.loads((gate/'execution.json').read_text())['complete']:run(gate,wall_seconds=10800)
    validate_chain(gate,complete=True);proof=audit(gate)
    (graph/'gate-audit.json').write_text(json.dumps(proof,indent=2)+'\n')
    suite=ET.Element('testsuite',name='Actual save-restore convolution to FC gate',tests='1',failures='0',errors='0',skipped='0')
    ET.SubElement(suite,'testcase',name='actual-output-and-counters-after-model-restore')
    ET.ElementTree(suite).write(graph/'gate-junit.xml',encoding='utf-8',xml_declaration=True)
    (graph/'checked-prefixes.json').write_text('{}\n')
    execute(SimpleNamespace(root=graph,compiled_root=None,compile_only=True,savable=True,stem_channel_lanes=stem_lanes(args),
                            function_metrics=bool(function_metrics(args)),postops_pipelined=bool(postops_pipeline(args))))


def checked_prefixes(root):
    checked=json.loads((root/'checked-prefixes.json').read_text())
    if not(root/'resume-chain.json').exists():
        require(checked=={},'Checked stages require actual execution ancestry')
        return checked
    chain=validate_chain(root)
    require(sorted(map(int,checked))==list(range(len(checked)))and len(checked)<=21,'Invalid checked stage prefix')
    plan=graph_plan(root);previous=-1
    for key,proof in checked.items():
        phase=int(key);index=proof['chunk_index']
        require(type(index)is int and previous<index<len(chain['chunks'])and
                proof['manifest_sha256']==chain['chunks'][index]['manifest_sha256'],'Unbound checked prefix ancestry')
        previous=index;report=proof['audit']
        if phase<20:
            require(report['result']=='PASS_CHECKED_CLOCKED_PREFIX_NOT_NETWORK'and
                    report['stop_after_instruction']==plan[phase]['stop_instruction']and
                    report['preparation_sha256']==chain['preparation_sha256']and
                    report['binary_sha256']==chain['binary_sha256']and
                    report['observed_prefix_cycles']==chain['chunks'][index]['end_status']['observed_cycles']and
                    report['real_rtl_junit']==1 and report['full_network_rtl_pass']is False and
                    report['full_network_cycles']is None,'Different checked prefix source, cycles or scope')
        else:
            require(report['result']=='EXECUTION_COMPLETE_AWAITING_FULL_AUDIT'and
                    chain['chunks'][index]['end_status']['status']=='COMPLETE_REQUIRES_INDEPENDENT_AUDIT',
                    'Final execution is incomplete')
    return checked


def publish_progress(args,phase,status,checkpoint,cipher_hash,checked,failure=None):
    args.public.mkdir(parents=True,exist_ok=True)
    report=dict(result='CHECKED_PREFIX_NOT_NETWORK_PASS'if checked else'UNVERIFIED_EXECUTION_SNAPSHOT_NOT_STAGE_PASS',
        profile=args.profile,stem_channel_lanes=stem_lanes(args),function_metrics=function_metrics(args),
        postops_pipelined=postops_pipeline(args),phase=args.phase,phase_name=phase['name'],slot=args.slot,
        phase_execution_complete=status,checkpoint_sha256=cipher_hash,numeric_prefix_checked=checked,
        full_network_rtl_pass=False,full_network_cycles=None)
    if failure is not None:
        report['result']='FAIL_NUMERIC_PREFIX_AUDIT'
        for key in('failure_code','failure_instruction','failure_word_offset'):
            if key in failure:report[key]=failure[key]
    (args.public/'progress.json').write_text(json.dumps(report,indent=2)+'\n')
    output=os.environ.get('GITHUB_OUTPUT')
    require(output is not None,'Actions output path required')
    with Path(output).open('a')as stream:
        stream.write('phase_complete='+str(status).lower()+'\ncheckpoint_name='+checkpoint+'\n')


def stage(args):
    key_from_environment()
    require(0<args.wall_seconds<=10800,'Inference wall budget must stay below the job checkpoint reserve')
    release=json.loads(args.release.read_text());ctx=context(args,release);root=args.workspace/'graph'
    plan=graph_plan(root);require(0<=args.phase<len(plan)and args.slot in(0,1,2),'Invalid stage or continuation slot')
    phase=plan[args.phase];remote_name=remote(args,ctx);checkpoint=choose_checkpoint(args,remote_name)
    if checkpoint:fetch(checkpoint,remote_name,args,ctx)
    else:
        require(args.phase==args.slot==0,'Fresh state allowed only at stem start')
        prepare_initial(args)
    verify_stem_policy(args,root);gate_proof(root,release,args.profile,postops_pipeline(args))
    checked=checked_prefixes(root)
    require(len(checked)<=args.phase+1,'Future stage proof in current execution')
    if str(args.phase)in checked:
        validate_chain(root);publish_progress(args,phase,True,checkpoint,sha(args.workspace.parent/'restored-clocked-state.aesgcm'),args.phase<20);return
    require(len(checked)==args.phase,'Cannot execute a stage before checking its predecessor')
    def phase_done(status):
        if args.phase==20:return status['status']=='COMPLETE_REQUIRES_INDEPENDENT_AUDIT'
        if args.phase==0:return status['stem_values']==200704 and status['engine_cycles']==0
        return status['last_logged_instruction']==phase['stop_instruction']
    latest=validate_chain(root)['chunks'][-1]if (root/'resume-chain.json').exists()else None
    if latest and phase_done(latest['end_status']):
        result=dict(status=latest['end_status'],chunk_index=latest['index'])
    else:
        result=run(root,stop_ip=None if args.phase in(0,20)else phase['stop_instruction'],
            stop_stem=args.phase==0,wall_seconds=args.wall_seconds)
    status=result['status'];done=phase_done(status);audit_error=None
    if done:
        try:
            if args.phase<20:
                from fpga.openeye.audit_clocked_graph import audit
                report=audit(SimpleNamespace(root=root,bundle_root=args.workspace/'bundle',stop_after_instruction=phase['stop_instruction']))
                require(report['result']=='PASS_CHECKED_CLOCKED_PREFIX_NOT_NETWORK','Uncertified prefix')
            else:report=dict(result='EXECUTION_COMPLETE_AWAITING_FULL_AUDIT',full_network_rtl_pass=False,full_network_cycles=None)
            checked[str(args.phase)]=dict(chunk_index=result['chunk_index'],manifest_sha256=validate_chain(root)['chunks'][-1]['manifest_sha256'],audit=report)
            (root/'checked-prefixes.json').write_text(json.dumps(checked,indent=2)+'\n')
        except Exception as error:
            audit_error=error
    name=f'phase-{args.phase:02d}-slot-{args.slot}-attempt-{os.environ["GITHUB_RUN_ATTEMPT"]}.aesgcm'
    output=args.workspace.parent/name;digest=seal(root,output,ctx,key_from_environment())
    response=subprocess.run(['rclone','copyto','--immutable','--config',str(args.config),str(output),remote_name+'/'+name],capture_output=True,text=True)
    require(response.returncode==0,'Private authenticated snapshot upload failed')
    from fpga.openeye.audit_clocked_graph import failure_report
    publish_progress(args,phase,done and audit_error is None,name,digest,done and audit_error is None and args.phase<20,
                     failure_report(audit_error)if audit_error is not None else None)
    if audit_error is not None:raise ValueError('Numeric prefix audit failed; exact private failed state preserved without stage PASS')from audit_error
    if not done and args.slot==2:raise ValueError('Stage saved before time budget; rerun failed jobs continues this exact snapshot')


def final(args):
    key_from_environment()
    release=json.loads(args.release.read_text());ctx=context(args,release);remote_name=remote(args,ctx)
    require(args.checkpoint and re.fullmatch(NAME,args.checkpoint),'Missing final authenticated execution state')
    fetch(args.checkpoint,remote_name,args,ctx);root=args.workspace/'graph';validate_chain(root,complete=True)
    verify_stem_policy(args,root);gate_proof(root,release,args.profile,postops_pipeline(args))
    checked=checked_prefixes(root);require(sorted(map(int,checked))==list(range(21)),'Incomplete21-phase execution')
    for i in range(20):require(checked[str(i)]['audit']['result']=='PASS_CHECKED_CLOCKED_PREFIX_NOT_NETWORK','Unchecked prior stage')
    args.public.mkdir(parents=True,exist_ok=True)
    (args.public/'gate').mkdir();shutil.copyfile(root/'gate-audit.json',args.public/'gate/audit.json');shutil.copyfile(root/'gate-junit.xml',args.public/'gate/junit.xml')
    shutil.copyfile(args.workspace/'ingress-public.json',args.public/'ingress.json')
    subprocess.run([sys.executable,str(ROOT/'fpga/openeye/audit_clocked_graph.py'),'--root',str(root),'--bundle-root',str(args.workspace/'bundle'),'--output',str(args.public/'network')],check=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('stage','final'))
    for name in('release','workspace','config','public'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--profile',choices=('dense','mac20','optimized'),required=True)
    parser.add_argument('--phase',type=int,default=20);parser.add_argument('--slot',type=int,default=0)
    parser.add_argument('--checkpoint',default='');parser.add_argument('--wall-seconds',type=int,default=10800)
    parser.add_argument('--stem-channel-lanes',type=int,choices=(1,3),default=1)
    parser.add_argument('--function-metrics',type=int,choices=(0,1),default=0)
    parser.add_argument('--postops-pipelined',type=int,choices=(0,1),default=0)
    args=parser.parse_args();stage(args)if args.mode=='stage'else final(args)
