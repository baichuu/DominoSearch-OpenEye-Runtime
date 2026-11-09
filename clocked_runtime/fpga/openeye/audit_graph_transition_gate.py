#!/usr/bin/env python3
"""Audit the actual convolution→FC gate, never a whole-network PASS."""
import argparse
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET
import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from fpga.openeye.validate_graph_program import validate,require,sha,read_words,PIN
from fpga.openeye.audit_clocked_graph import parse_timeline,signed


def audit(root):
    root=Path(root);prep=json.loads((root/'preparation.json').read_text())
    require(prep['result']=='OFFLINE_TRANSITION_GATE_NOT_NETWORK'and prep['gate_kind']=='conv-postops-to-native-fc','Wrong protocol gate scope')
    validate(root,whole=False)
    identity=json.loads((root/'clocked-build/identity.json').read_text());execution=json.loads((root/'execution.json').read_text())
    require(identity['openeye_pin']==PIN and identity['parameters']['USE_STEM']==0 and
            identity['preparation_sha256']==sha(root/'preparation.json')and execution['returncode']==0 and
            execution['identity_sha256']==sha(root/'clocked-build/identity.json'),'Failed/foreign protocol execution')
    require(sha(root/'clocked-build/obj/Vopeneye_graph_tb')==identity['binary_sha256'],'Changed actual binary')
    if identity.get('driver_protocol')=='verilated-saveable-low-edge-v1':
        from fpga.openeye.clocked_resume import validate_chain
        require(execution['resume_chain_sha256']==sha(root/'resume-chain.json'),'Changed actual gate resume chain')
        validate_chain(root,complete=True)
    for name,digest in{**identity['sources'],**execution['files'],**prep['oracle_files']}.items():
        require(not Path(name).is_absolute()and '..'not in Path(name).parts and sha(root/name)==digest,'Changed source/oracle/actual trace')
    require(read_words(root/'actual-output.hex',16)==read_words(root/'golden-output.hex',16),'Native FC output differs after convolution')
    require(read_words(root/'actual-raw.hex',16)==read_words(root/'golden-raw.hex',16),'Raw convolution/FC output differs, including padding')
    rows=[line.split()for line in(root/'actual-forwarded.hex').read_text().splitlines()]
    require(all(len(v)==3 and all(re.fullmatch('[0-9a-fA-F]{'+str(n)+'}',t)for t,n in zip(v,(8,8,16)))for v in rows),'Malformed actual forwarded trace')
    expected_conv=read_words(root/'golden-conv.hex',16);expected_fc=read_words(root/'golden-output.hex',16)
    expected=[(1,prep['conv_output_base']+i,v)for i,v in enumerate(expected_conv)]+[(3,prep['output_base']+i,v)for i,v in enumerate(expected_fc)]
    require([tuple(int(v,16)for v in row)for row in rows]==expected,'Incorrect actual conv/post-op/FC forwarding')
    program=[[int(line[i:i+8],16)for i in range(0,len(line),8)][::-1]for line in(root/'program.hex').read_text().splitlines()]
    actual=read_words(root/'actual-accepted.hex',16);static=read_words(root/'packets.hex',16);memory=signed(read_words(root/'initial-memory.hex',16))
    for ip in(0,2):
        row=program[ip];base,count,offset,n=row[1:5]
        if ip==0:
            a=memory[:64*prep['size']**2].reshape(64,prep['size'],prep['size']);used=row[21]
            values=[int(a[c+k,x,y])for c in range(0,64,used)for y in range(prep['size'])for x in range(prep['size'])for k in range(used)]
        else:values=memory[row[9]+448:row[9]+512].tolist()
        words=[sum(int(v)<<(8*k)for k,v in enumerate(values[i:i+8]))for i in range(0,len(values),8)]
        live=[v for i in range(0,len(words),8)for v in words[i:i+8][::-1]]+[0]*(n-len(words))
        require(len(live)==n and all(v==0 for v in static[base+offset:base+offset+n]),'Unproven live activation field')
        static[base+offset:base+offset+n]=live
    require(actual==static,'Actual native core input differs from model/live SRAM data')
    starts,ends,states,intervals=parse_timeline((root/'simulation.log').read_text(),program)
    marker=re.findall(r'GRAPH_OUTPUT_DONE values=(\d+) latency=(\d+) slots=(\d+) engine_cycles=(\d+)',(root/'simulation.log').read_text())
    require(len(marker)==1 and int(marker[0][0])==8 and int(marker[0][2])==int(marker[0][1])+1,'Missing actual protocol-gate timeline')
    return dict(result='PASS_ACTUAL_CONV_TO_NATIVE_FC_PROTOCOL_GATE',openeye_pin=PIN,optimized=prep['optimized'],
        postops_pipelined=bool(identity['parameters'].get('POSTOPS_PIPELINED',0)),
        input_size=prep['size'],conv_filters=prep['filters'],checked_convolution_postops_values=len(expected_conv),
        checked_fc_logits=8,checked_raw_words=len(read_words(root/'actual-raw.hex',16)),accepted_words=len(actual),
        latency_cycles=int(marker[0][1]),binary_sha256=identity['binary_sha256'],preparation_sha256=sha(root/'preparation.json'),
        actual_raw_sha256=sha(root/'actual-raw.hex'),actual_forwarded_sha256=sha(root/'actual-forwarded.hex'),
        real_rtl_junit=1,full_network_cycles=None,full_network_rtl_pass=False,accuracy=False,board=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in('root','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();require(not args.output.exists(),'New gate evidence directory required');args.output.mkdir()
    try:report=audit(args.root)
    except Exception as error:
        report=dict(result='FAIL_ACTUAL_PROTOCOL_GATE',failure_type=type(error).__name__,full_network_cycles=None,full_network_rtl_pass=False)
        suite=ET.Element('testsuite',name='Actual conv to FC protocol gate',tests='1',failures='1',errors='0',skipped='0')
        case=ET.SubElement(suite,'testcase',name='persistent-core-mode-switch');ET.SubElement(case,'failure',message='Actual RTL/source/stream audit failed')
        ET.ElementTree(suite).write(args.output/'junit.xml',encoding='utf-8',xml_declaration=True)
        (args.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n');raise
    (args.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    suite=ET.Element('testsuite',name='Actual conv to FC protocol gate',tests='1',failures='0',errors='0',skipped='0')
    ET.SubElement(suite,'testcase',name='persistent-core-mode-switch',classname='whole-network-int8-fixedpoint-v1-gate')
    ET.ElementTree(suite).write(args.output/'junit.xml',encoding='utf-8',xml_declaration=True);print(json.dumps(report,indent=2))
