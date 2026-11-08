#!/usr/bin/env python3
"""Fail-closed structural ingress for the separate whole-network RTL experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import re

PIN='9a742fbd42a55ebfa8fcd39aa41dfa36a367572f'
LAYERS=[]
for group in range(1,5):
    for block in range(2):
        LAYERS.extend([f'layer{group}.{block}.conv1',f'layer{group}.{block}.conv2'])
        if group>1 and block==0:LAYERS.append(f'layer{group}.{block}.downsample.0')
LAYERS.append('fc')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok,message):
    if not ok:raise ValueError(message)


def read_words(path,digits):
    tokens=path.read_text().splitlines()
    require(tokens and all(re.fullmatch('[0-9a-fA-F]{'+str(digits)+'}',v)for v in tokens),'Malformed program/ROM words')
    return[int(v,16)for v in tokens]


def validate(root,*,whole=True):
    root=Path(root).resolve();prep=json.loads((root/'preparation.json').read_text())
    for name,digest in prep['files'].items():
        require(Path(name).name==name and not(root/name).is_symlink()and sha(root/name)==digest,'Changed/unsafe graph model input')
    program=[[int(line[i:i+8],16)for i in range(0,len(line),8)][::-1]
             for line in(root/'program.hex').read_text().splitlines()]
    require(program and all(len(row)==32 for row in program),'Wrong instruction format')
    require(program[-1][0]==0 and all(row[0]in(1,2,3,4,5)for row in program[:-1]),'Unexpected/early HALT instruction')
    require(len(program)==prep['instructions'],'Wrong instruction count')
    memory=prep['memory_values'];packet_limit=prep['packet_words'];map_limit=prep['map_lanes'];coeff_limit=prep['coefficient_count']
    require(0<prep['initial_values']<=memory and 0<prep['output_values']and
            0<=prep['output_base']<prep['output_base']+prep['output_values']<=memory,'Invalid initial/output scratch allocation')
    raw_map=read_words(root/'maps.hex',8);require(len(raw_map)==map_limit,'Wrong map ROM length')
    require(sum(1 for _ in(root/'packets.hex').open())==packet_limit,'Wrong packet ROM length')
    require(sum(1 for _ in(root/'initial-memory.hex').open())==prep['initial_values'],'Wrong actual input length')
    coefficients=(root/'coefficients.hex').read_text().splitlines()
    require(len(coefficients)==coeff_limit and all(re.fullmatch('[0-9a-fA-F]{16} [0-9a-fA-F]{16}',v)for v in coefficients),'Wrong coefficient ROM')
    next_packet=0;commands=[];op_counts={str(op):0 for op in range(1,6)}
    for index,row in enumerate(program[:-1]):
        op=row[0];op_counts[str(op)]+=1
        if op==1:
            _,base,count,iact,iact_words,map_base,raw_words,acc_base,group,src,sw,sh,channels,c,tw,th,xstart,cw,ch,raster,decimate,used,*_=row
            require(base==next_packet and count>0 and base+count<=packet_limit,'Discontinuous/missing packet bytes')
            next_packet+=count
            require(0<=iact<=iact+iact_words<=count and (iact_words==0 or iact_words%8==0),'Invalid activation section')
            require(raw_words>0 and raw_words<=prep['raw_words']and 0<=map_base<map_base+2*raw_words<=map_limit,'Output frame/map exceeds declared capacity')
            selected=raw_map[map_base:map_base+2*raw_words];valid=[v for v in selected if v!=0xffffffff]
            require(len(valid)==len(set(valid))and all(acc_base+v<memory for v in valid),'Duplicate/out-of-range scatter mapping')
            require(0<sw<=56 and 0<sh<=56 and 0<tw<=56 and 0<th<=56 and
                    channels>0 and channels%64==0 and c%64==0 and c+64<=channels and
                    (group==c//64 if whole else 0<=group<channels//64)and
                    src+channels*sw*sh<=memory and used in(4,8)and raster==decimate==0 and
                    xstart+cw<=sw and ch<=sh,'Invalid dynamic activation mapping')
            require(iact_words==0 or iact_words>=64*tw*th//8,'Truncated live activation payload')
            commands.append(index)
        else:
            _,src,dst,count,coefficient,pixels,flags,shift,*_=row
            require(count>0 and src+count<=memory,'Invalid scan source range')
            if op==4:require(0<coefficient<=65535 and count%coefficient==0,'Invalid GAP pixel count')
            output_count=count//coefficient if op==4 else count
            require(dst+output_count<=memory,'Invalid scan output range')
            if op==2:
                require(pixels>0 and count%pixels==0 and coefficient+count//pixels<=coeff_limit and flags in(0,1)and shift<=40,'Invalid affine geometry/coefficients')
            if op==3:
                require(flags+count<=memory and coefficient<coeff_limit and shift<=40,'Invalid residual skip/scale')
            if op==4:
                require(0<coefficient<=65535 and count%coefficient==0,'Invalid GAP pixel count')
            if op==5:require(coefficient+count<=coeff_limit,'Invalid FC bias/scale coverage')
    require(next_packet==packet_limit,'Unused/unaccounted model command bytes')
    if whole:
        require(prep['result']=='OFFLINE_COMPLETE_GRAPH_PREPARED_NOT_DUT_EVIDENCE'and prep['connected_stem']is True and
                prep['experiment']=='whole-network-int8-fixedpoint-v1'and prep['initial_values']==150528 and
                prep['output_values']==1000,'Complete image-to-logit graph required')
        require([s['layer']for s in prep['stages']]==LAYERS,'Missing/reordered/duplicate source stage')
        require(op_counts['2']==19,'All19 backbone affine operations required')
        require(op_counts['3']==8 and op_counts['4']==1 and op_counts['5']==1,'Missing residual/GAP/FC bias operators')
        stage_commands=[c['instruction']for stage in prep['stages']for c in stage['commands']]
        require(stage_commands==commands and len(commands)==prep['commands'],'Incomplete layer/command attribution')
        for stage in prep['stages']:
            require(stage['first_instruction']==stage['commands'][0]['instruction']and
                    stage['last_instruction']==stage['commands'][-1]['instruction']+1,'Invalid stage interval')
            require(program[stage['last_instruction']][0]==(5 if stage['layer']=='fc'else 2),'Stage omitted post-processing')
    return dict(result='VALID_STRUCTURAL_GRAPH_NOT_DUT_PASS',instructions=len(program),commands=len(commands),
        op_counts=op_counts,connected_stem=prep.get('connected_stem',False),model=prep.get('model'),optimized=prep.get('optimized',False),
        preparation_sha256=sha(root/'preparation.json'),inference_input_values=prep['initial_values'],
        full_network_rtl_pass=False,full_network_cycles=None)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(validate(args.root),indent=2))
