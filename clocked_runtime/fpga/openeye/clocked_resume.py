#!/usr/bin/env python3
"""Execute authenticated, cycle-continuous chunks of one saveable RTL model.

Chunk completion is process progress, not layer/network correctness. Independent
prefix/final audits are required before publishing any numeric PASS.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from fpga.openeye.validate_graph_program import require,PIN

TRACE_NAMES=('actual-accepted.hex','actual-raw.hex','actual-forwarded.hex','actual-output.hex','actual-stem.hex','simulation.log')
PROTOCOL='verilated-saveable-low-edge-v1'


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for block in iter(lambda:stream.read(1<<20),b''):value.update(block)
    return value.hexdigest()


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def abi():return dict(system=platform.system(),machine=platform.machine(),libc=list(platform.libc_ver()))


def graph_plan(root):
    prep=json.loads((root/'preparation.json').read_text())
    program=[int(v,16)&0xffffffff for v in(root/'program.hex').read_text().splitlines()]
    phases=[dict(index=0,name='stem',stop_stem=True,stop_instruction=-1)]
    for stage in prep['stages']:
        end=stage['last_instruction']
        if end+1<len(program)and program[end+1]==3:end+=1
        phases.append(dict(index=len(phases),name=stage['layer'],stop_stem=False,stop_instruction=end))
    require(len(phases)==21 and phases[-1]['name']=='fc','Complete21-phase image-to-logit program required')
    return phases


def identity(root):
    prep=json.loads((root/'preparation.json').read_text());build=json.loads((root/'clocked-build/identity.json').read_text())
    require(build.get('driver_protocol')==PROTOCOL and build['openeye_pin']==PIN and
            build['preparation_sha256']==sha(root/'preparation.json'),'Different saveable model/preparation')
    require(sha(root/'clocked-build/obj/Vopeneye_graph_tb')==build['binary_sha256'],'Changed saveable binary')
    for name,digest in {**build['sources'],**prep['files']}.items():
        p=Path(name);require(not p.is_absolute()and '..'not in p.parts and not(root/p).is_symlink()and sha(root/p)==digest,'Changed model/source/ROM')
    return build


def validate_chain(root,*,complete=False):
    build=identity(root);document=json.loads((root/'resume-chain.json').read_text());entries=document['chunks']
    require(document['protocol']==PROTOCOL and document['binary_sha256']==build['binary_sha256']and
            document['preparation_sha256']==sha(root/'preparation.json')and document['abi']==abi(),'Foreign model snapshot ABI/source')
    require(entries and [v['index']for v in entries]==list(range(len(entries))),'Missing/duplicate snapshot chunk')
    previous=None
    for entry in entries:
        unsigned=dict(entry);digest=unsigned.pop('manifest_sha256')
        require(hashlib.sha256(canonical(unsigned)).hexdigest()==digest,'Changed snapshot chunk manifest')
        require(entry['parent_manifest_sha256']==(previous['manifest_sha256']if previous else None),'Broken snapshot ancestry')
        require(entry['start_status']==(previous['end_status']if previous else None),'Different restored cycle/counter state')
        if previous:
            a,b=previous['end_status'],entry['end_status']
            require(b['absolute_cycles']>a['absolute_cycles']and b['simulation_time']>a['simulation_time']and
                    b['observed_cycles']>=a['observed_cycles']and b['engine_cycles']>=a['engine_cycles'],'Reset/backward progress after restore')
            require(b['simulation_time']-a['simulation_time']==10000*(b['absolute_cycles']-a['absolute_cycles']),
                    'Restored model clock phase/time does not match actual edge counters')
        previous=entry
    latest=entries[-1]
    require(set(latest['files'])==set(TRACE_NAMES)and json.loads((root/'resume-status.json').read_text())==latest['end_status'],
            'Missing trace stream or different latest model counters')
    for name,record in latest['files'].items():
        require(name in TRACE_NAMES and(root/name).stat().st_size==record['bytes']and sha(root/name)==record['sha256'],'Changed checkpoint trace prefix')
    if latest['end_status']['status']=='PAUSED_NOT_NETWORK_PASS':
        name=latest['snapshot'];p=Path(name)
        require(not p.is_absolute()and '..'not in p.parts and p.parts[0]=='resume-chunks'and
                sha(root/p)==latest['snapshot_sha256'],'Changed/missing actual model snapshot')
    if complete:require(latest['end_status']['status']=='COMPLETE_REQUIRES_INDEPENDENT_AUDIT','Incomplete model execution')
    return document


def run(root,*,stop_ip=None,stop_stem=False,stop_edges=None,wall_seconds=12600):
    root=Path(root).resolve();require(ROOT not in root.parents and root!=ROOT,'Private inference root required')
    build=identity(root);chain_path=root/'resume-chain.json';resume=chain_path.exists()
    require(wall_seconds>0 and(stop_edges is None or stop_edges>0)and
            (stop_ip is None or 0<=stop_ip<build['parameters']['INSTRUCTIONS']-1),'Invalid continuation stop boundary')
    if resume:
        chain=validate_chain(root);latest=chain['chunks'][-1];start_status=latest['end_status']
        require(start_status['status']=='PAUSED_NOT_NETWORK_PASS','Completed model cannot be executed again')
        snapshot=root/latest['snapshot']
    else:
        require(not any((root/n).exists()for n in(*TRACE_NAMES,'execution.json','resume-status.json')),'Never overwrite existing simulation output')
        chain=dict(protocol=PROTOCOL,binary_sha256=build['binary_sha256'],preparation_sha256=sha(root/'preparation.json'),abi=abi(),chunks=[])
        start_status=None;snapshot=None
    index=len(chain['chunks']);chunks=root/'resume-chunks';chunks.mkdir(exist_ok=True,mode=0o700)
    model_file=chunks/f'chunk-{index:04d}.model'
    require(not model_file.exists(),'Fresh model snapshot path required')
    args=[str(root/'clocked-build/obj/Vopeneye_graph_tb'),'--save',str(model_file),'--wall-seconds',str(wall_seconds),
        '+PROGRAM=program.hex','+PACKETS=packets.hex','+MAP=maps.hex','+COEFFICIENTS=coefficients.hex','+INITIAL=initial-memory.hex']
    if build['parameters']['USE_STEM']:args+=['+STEM_WEIGHTS=stem-weights.hex','+STEM_COEFFICIENTS=stem-coefficients.hex']
    if snapshot:args+=['--resume',str(snapshot)]
    if stop_ip is not None:args+=['--stop-ip',str(stop_ip)]
    if stop_stem:args+=['--stop-stem']
    if stop_edges:args+=['--stop-after-edges',str(stop_edges)]
    started=time.perf_counter_ns()
    with(root/'simulation.log').open('ab'if resume else'xb')as log:
        result=subprocess.run(args,cwd=root,stdout=log,stderr=subprocess.STDOUT,timeout=wall_seconds+300)
    result.check_returncode();status=json.loads((root/'resume-status.json').read_text())
    require(status['status']in('PAUSED_NOT_NETWORK_PASS','COMPLETE_REQUIRES_INDEPENDENT_AUDIT'),'Unknown snapshot driver result')
    entry=dict(index=index,parent_manifest_sha256=chain['chunks'][-1]['manifest_sha256']if resume else None,
        start_status=start_status,end_status=status,host_wall_ns=time.perf_counter_ns()-started,
        snapshot=str(model_file.relative_to(root))if model_file.exists()else None,
        snapshot_sha256=sha(model_file)if model_file.exists()else None,
        files={name:dict(bytes=(root/name).stat().st_size,sha256=sha(root/name))for name in TRACE_NAMES})
    entry['manifest_sha256']=hashlib.sha256(canonical(entry)).hexdigest();chain['chunks'].append(entry)
    with(chunks/f'chunk-{index:04d}.json').open('x')as out:out.write(json.dumps(entry,indent=2)+'\n')
    chain_path.write_text(json.dumps(chain,indent=2)+'\n')
    execution=dict(returncode=0,complete=status['status']=='COMPLETE_REQUIRES_INDEPENDENT_AUDIT',
        host_simulation_wall_ns=sum(v['host_wall_ns']for v in chain['chunks']),identity_sha256=sha(root/'clocked-build/identity.json'),
        resume_chain_sha256=sha(chain_path),files={name:record['sha256']for name,record in entry['files'].items()})
    (root/'execution.json').write_text(json.dumps(execution,indent=2)+'\n');validate_chain(root,complete=execution['complete'])
    return dict(status=status,chunk_index=index,complete=execution['complete'],full_network_rtl_pass=False,full_network_cycles=None)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--stop-ip',type=int);parser.add_argument('--stop-stem',action='store_true')
    parser.add_argument('--stop-after-edges',type=int);parser.add_argument('--wall-seconds',type=int,default=12600)
    args=parser.parse_args();print(json.dumps(run(args.root,stop_ip=args.stop_ip,stop_stem=args.stop_stem,stop_edges=args.stop_after_edges,wall_seconds=args.wall_seconds),indent=2))
