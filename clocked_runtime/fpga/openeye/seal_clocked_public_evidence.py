#!/usr/bin/env python3
"""Publish a small explicit evidence allowlist, never simulator logs or operands."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET

COMMON={'result','full_network_cycles','full_network_rtl_pass','failure_type',
        'failure_code','failure_instruction','failure_word_offset'}
FIELDS={
 'ingress.json':COMMON|{'profile','model','optimized','archive_sha256','graph_preparation_sha256','gate_preparation_sha256','commands','openeye_pin'},
 'gate/audit.json':COMMON|{'openeye_pin','optimized','input_size','conv_filters','checked_convolution_postops_values','checked_fc_logits',
     'checked_raw_words','accepted_words','latency_cycles','binary_sha256','preparation_sha256','actual_raw_sha256','actual_forwarded_sha256','real_rtl_junit','accuracy','board'},
 'network/audit.json':COMMON|{'experiment','model','optimized','openeye_pin','source_checkpoint_sha256','source_scheme_sha256',
     'arithmetic_contract_sha256','input_sha256','preparation_sha256','binary_sha256','checked_stem_values','checked_forwarded_values','checked_logits',
     'stem_components','accepted_input_words','checked_raw_words','stages','postops','stem_latency_cycles','observation_slots','output_sha256',
     'real_rtl_junit','native_core_states','clock_period_ns','clock_scope','interval','activation_measurement','model_initialization','hardware_scope',
     'accuracy','synthesis','post_route','board','host','resume_chain_sha256','resumed_chunks',
     'verification_vectors','external_memory_model','latency_scope','stem_channel_lanes'}}
ROW_FIELDS={
 'stages':{'layer','scheme','commands','checked_values','latency_cycles','convolution_dma_accumulation_cycles','postops_cycles','output_sha256'},
 'postops':{'operation','site','checked_values','latency_cycles'},
 'native_core_states':{'instruction','state','cycles','input_words'},
 'stem_components':{'operation','accepted_values','first_output_edge','last_output_edge','output_span_cycles','scope'}}


def require(ok,message):
    if not ok:raise ValueError(message)


def seal(root,output,profile,commit,run_id):
    require(profile in ('dense','mac20','optimized'),'Unknown profile')
    require(re.fullmatch('[0-9a-f]{40}',commit)and re.fullmatch('[0-9]+',run_id),'Invalid execution identity')
    require(not output.exists()and root.resolve()!=output.resolve(),'Fresh artifact directory required')
    destination=output;output=output.with_name(output.name+'.partial')
    require(not output.exists(),'Fresh evidence staging directory required')
    output.mkdir(parents=True,mode=0o700)
    copied=[];network_pass=False
    for relative in ('ingress.json','gate/audit.json','network/audit.json'):
        path=root/relative
        if not path.exists():continue
        require(path.is_file()and not path.is_symlink()and path.stat().st_size<16_000_000,'Invalid public evidence input')
        data=json.loads(path.read_text())
        require(isinstance(data,dict)and set(data)<=FIELDS[relative],'Unknown public evidence fields')
        if 'failure_code'in data:
            require(data['failure_code']in {'RAW_OUTPUT_MISMATCH','INPUT_PACKET_MISMATCH','FORWARDED_VALUE_MISMATCH'}and
                    all(type(data.get(k))is int and data[k]>=0 for k in('failure_instruction','failure_word_offset')),
                    'Unallowlisted audit failure diagnostic')
            require(data.get('full_network_rtl_pass')is False and data.get('full_network_cycles')is None,
                    'Numeric failure cannot certify a network result')
        for key,value in data.items():
            if key in ROW_FIELDS:
                require(isinstance(value,list)and value and all(isinstance(row,dict)and set(row)==ROW_FIELDS[key]for row in value),'Unexpected public table schema')
                require(all(not isinstance(item,(dict,list))for row in value for item in row.values()),'Tensor-like data in public table')
            else:require(not isinstance(value,(list,dict)),'Unexpected tensor-like public value')
        # Auditors expose numeric counters, hashes and fixed schema labels only.
        # Recheck every scalar before copying, including nested rows.
        def check(value):
            if value is None or isinstance(value,(bool,int,float)):return
            if isinstance(value,list):
                for item in value:check(item)
            elif isinstance(value,dict):
                require(all(re.fullmatch('[a-zA-Z0-9_]+',k)for k in value),'Unsafe public evidence key')
                for item in value.values():check(item)
            else:
                require(isinstance(value,str)and len(value)<512 and not any(c in value for c in ('\n','\r','\\','/','@','{','}')),
                        'Private/free-form content in public evidence')
        check(data)
        if relative=='network/audit.json':
            network_pass=data.get('result')=='PASS_INDEPENDENT_COMPLETE_CLOCKED_NETWORK_AUDIT'
            require(network_pass==bool(data.get('full_network_rtl_pass')),'Inconsistent network result')
            require(network_pass or data.get('full_network_cycles')is None,'Uncertified network cycle total')
        target=output/relative;target.parent.mkdir(exist_ok=True)
        target.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');copied.append(relative)
    for scope in ('gate','network'):
        path=root/scope/'junit.xml'
        if not path.exists():continue
        require(not path.is_symlink()and path.stat().st_size<200_000,'Invalid JUnit input')
        suite=ET.parse(path).getroot();cases=list(suite.iter('testcase'))
        require(suite.tag=='testsuite'and len(cases)==1 and suite.get('tests')=='1','Unexpected audit JUnit shape')
        failed=bool(cases[0].findall('failure')or cases[0].findall('error'))
        skipped=bool(cases[0].findall('skipped'))
        require(int(suite.get('failures','0'))+int(suite.get('errors','0'))==int(failed)and int(suite.get('skipped','0'))==int(skipped),
                'Inconsistent real audit JUnit status')
        if scope=='network':require(network_pass==(not failed and not skipped),'Network report and real JUnit disagree')
        clean=ET.Element('testsuite',name='Independent actual '+scope+' RTL audit',tests='1',
                         failures=str(int(failed)),errors='0',skipped=str(int(skipped)))
        case=ET.SubElement(clean,'testcase',name='actual-output-source-and-counter-check')
        if failed:ET.SubElement(case,'failure',message='Independent actual RTL evidence audit failed')
        if skipped:ET.SubElement(case,'skipped',message='Actual evidence audit did not complete')
        target=output/scope/'junit.xml';target.parent.mkdir(exist_ok=True)
        ET.ElementTree(clean).write(target,encoding='utf-8',xml_declaration=True);copied.append(scope+'/junit.xml')
    require(not network_pass or 'network/junit.xml'in copied,'Network PASS requires actual audit JUnit')
    columns={
        'layers.csv':('layer','scheme','commands','checked_values','latency_cycles','convolution_dma_accumulation_cycles','postops_cycles','output_sha256'),
        'postops.csv':('operation','site','checked_values','latency_cycles'),
        'native-core-states.csv':('instruction','state','cycles','input_words')}
    if network_pass:
        for name,header in columns.items():
            path=root/'network'/name
            require(path.is_file()and not path.is_symlink()and path.stat().st_size<32_000_000,'Missing/invalid certified cycle table')
            with path.open(newline='')as stream:
                reader=csv.DictReader(stream);require(tuple(reader.fieldnames or ())==header,'Unexpected cycle table fields')
                rows=list(reader)
            require(rows and all(set(row)==set(header)and all(isinstance(v,str)and re.fullmatch('[a-zA-Z0-9_.:-]+',v)for v in row.values())for row in rows),
                    'Unallowlisted cycle table contents')
            target=output/'network'/name
            with target.open('w',newline='')as stream:
                writer=csv.DictWriter(stream,fieldnames=header);writer.writeheader();writer.writerows(rows)
            copied.append('network/'+name)
    metadata=dict(schema_version=1,profile=profile,execution_commit=commit,run_id=run_id,
                  full_network_rtl_pass=network_pass,
                  result='COMPLETE_AUDITED_NETWORK'if network_pass else'NO_CERTIFIED_COMPLETE_NETWORK_RESULT',
                  files={name:hashlib.sha256((output/name).read_bytes()).hexdigest()for name in copied})
    (output/'artifact-manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    output.rename(destination)
    return metadata


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','output'):parser.add_argument('--'+name,type=Path,required=True)
    for name in ('profile','execution-commit','run-id'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args();print(json.dumps(seal(args.root,args.output,args.profile,args.execution_commit,args.run_id),indent=2))
