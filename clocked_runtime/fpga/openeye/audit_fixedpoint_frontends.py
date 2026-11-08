#!/usr/bin/env python3
"""Read-only source, actual-operand, arithmetic and counter audit of stem RTL."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from fpga.openeye.prepare_packed_layer_fixture import decode_and_quantize,packed_layer,conv_iohw,sha256


def require(condition,message):
    if not condition:raise ValueError(message)


def signed_words(path,bits):
    tokens=path.read_text().splitlines();digits=bits//4
    require(all(re.fullmatch('[0-9a-fA-F]{'+str(digits)+'}',v)for v in tokens),'Unknown actual RTL word')
    words=[int(v,16)for v in tokens]
    return np.asarray([v-(1<<bits)if v>>(bits-1)else v for v in words],dtype=np.int64)


def independent_stem(image,weights,m,b):
    # Deliberately use plane-wise shift/add, not runtime's window/einsum oracle.
    require(image.shape==(3,224,224)and weights.shape==(3,64,7,7),'Wrong full stem geometry')
    padded=np.zeros((3,230,230),dtype=np.int64);padded[:,3:227,3:227]=image
    accumulator=np.zeros((64,112,112),dtype=np.int64)
    for c in range(3):
        for ky in range(7):
            for kx in range(7):
                plane=padded[c,kx:kx+224:2,ky:ky+224:2]
                accumulator+=weights[c,:,ky,kx,None,None]*plane[None,:,:]
    wide=accumulator*m[:,None,None]+b[:,None,None]
    quotient=np.floor_divide(wide,1<<24);remainder=wide-quotient*(1<<24)
    quotient+=((remainder>1<<23)|((remainder==1<<23)&((quotient&1)!=0)))
    post=np.clip(quotient,0,127)
    pooled=np.zeros((64,56,56),dtype=np.int64)
    p=np.zeros((64,114,114),dtype=np.int64);p[:,1:113,1:113]=post
    for dx in range(3):
        for dy in range(3):np.maximum(pooled,p[:,dx:dx+112:2,dy:dy+112:2],out=pooled)
    return pooled


def audit(root,bundle):
    root=Path(root);bundle=Path(bundle);ev=root/'evidence';private=root/'private'
    sums={line.split(None,1)[1].strip():line.split(None,1)[0]for line in(ev/'SHA256SUMS').read_text().splitlines()}
    require(set(sums)=={str(p.relative_to(ev))for p in ev.rglob('*')if p.is_file()and p.name!='SHA256SUMS'},'Incomplete frontend manifest')
    require(all(not Path(n).is_absolute()and'..'not in Path(n).parts and sha256(ev/n)==h for n,h in sums.items()),'Changed frontend evidence')
    comparison=json.loads((ev/'comparison.json').read_text());contract=json.loads((ev/'contract.json').read_text())
    require(comparison['result']=='PASS_MATCHED_INT8_STEM_FRONTENDS'and comparison['full_network_cycles']is None,'Wrong frontend claim')
    identity=json.loads((ev/'rtl/source-identity.json').read_text())
    require(identity['geometry']==dict(WIDTH=224,HEIGHT=224,CHANNELS=3,FILTERS=64,KERNEL=7,STRIDE=2,SHIFT=24),'Wrong common geometry')
    require(identity['simulator']=='verilator'and identity['openeye_pin']=='9a742fbd42a55ebfa8fcd39aa41dfa36a367572f','Wrong execution identity')
    require(sha256(private/'rtl/build/obj/Vopeneye_stem_frontend_tb')==identity['binary_sha256'],'Changed actual binary')
    for name,digest in identity['rtl_sha256'].items():
        p=private/'rtl'/('relu-source'if name=='af_cluster.v'else'build')/name
        require(sha256(p)==digest,'Changed actual stem source')
    unsigned_contract=dict(contract);expected=unsigned_contract.pop('contract_sha256')
    require(hashlib.sha256(json.dumps(unsigned_contract,sort_keys=True,separators=(',',':')).encode()).hexdigest()==expected==comparison['contract_sha256'],'Wrong arithmetic contract')
    vectors=bundle/'mac20/test-vectors-100';v=json.loads((vectors/'manifest.json').read_text());image_file=vectors/v['input']['file']
    require(sha256(image_file)==contract['vector_sha256']==v['input']['sha256'],'Changed input vector')
    source=np.memmap(image_file,dtype='<f4',mode='r',shape=(100,3,224,224))[contract['sample_index']].copy().astype(np.float64).transpose(0,2,1)
    image=np.clip(np.rint(source/contract['input_scale']),-127,127).astype(np.int64)
    rows=[]
    require([r['model']for r in comparison['rows']]==['dense','mac20'],'Wrong matched models')
    for index,row in enumerate(comparison['rows']):
        name=row['model'];run=private/'rtl'/f'command-{index:04d}';record=json.loads((ev/'rtl'/f'stem-{index:04d}.json').read_text())
        require(record==row['rtl_evidence']and record['site']==name+'.stem','Reordered/changed stem evidence')
        packed=bundle/name/'packed-v1';manifest=json.loads((packed/'manifest.json').read_text())
        require(sha256(packed/'manifest.json')==row['packed_manifest_sha256']and manifest['checkpoint']['sha256']==row['source_checkpoint_sha256'],'Changed stem model source')
        item=packed_layer(manifest,'conv1');values,q,_=decode_and_quantize(packed,item)
        weights=np.asarray(conv_iohw(values,item['source_weight_shape'],item['grouping']['grouped_order']))
        require(np.array_equal(signed_words(run/'image.hex',8).reshape(image.shape),image),'RTL image differs from fixed encoding')
        require(np.array_equal(signed_words(run/'weights.hex',8).reshape(weights.shape),weights),'RTL weights differ from immutable model')
        operands=[]
        for line in(run/'coefficients.hex').read_text().splitlines():
            tokens=line.split();require(len(tokens)==2 and all(re.fullmatch('[0-9a-fA-F]{16}',v)for v in tokens),'Bad coefficient words')
            words=[int(v,16)for v in tokens];operands.append([v-(1<<64)if v>>63 else v for v in words])
        coefficients=np.asarray(operands,dtype=np.int64);require(coefficients.shape==(64,2),'Wrong BN coverage')
        golden=independent_stem(image,weights,coefficients[:,0],coefficients[:,1])
        actual=signed_words(run/'output.hex',8).reshape(64,56,56)
        require(np.array_equal(actual,golden),'Independent full signed stem golden mismatch')
        require(np.array_equal(np.fromfile(private/(name+'-stem.int8'),dtype='i1').reshape(actual.shape),actual),'Forwarded stem boundary differs from actual RTL')
        require(sha256(private/(name+'-stem.int8'))==row['forwarded_int8_sha256'],'Forwarded stem hash differs')
        for filename,key in [('image.hex','private_input_sha256'),('weights.hex','private_weight_sha256'),('coefficients.hex','private_coefficients_sha256')]:require(sha256(run/filename)==record[key],'Actual operand hash differs')
        matches=re.findall(r'STEM_FRONTEND_DONE outputs=(\d+) latency=(\d+) slots=(\d+) stalls=(\d+)',(run/'simulation.log').read_text())
        require(len(matches)==1,'Missing raw stem counters');count,elapsed,slots,stalls=map(int,matches[0])
        require(count==record['checked_values']==200704 and elapsed==record['frontend_latency_cycles']and slots==elapsed+1==record['observation_slots']and stalls==record['output_stall_cycles'],'Wrong stem timeline')
        cases=list(ET.parse(ev/'rtl'/f'junit-{index:04d}.xml').iter('testcase'))
        require(len(cases)==1 and not any(cases[0].find(t)is not None for t in('error','failure','skipped')),'Failed/missing real stem JUnit')
        rows.append(dict(model=name,checked_values=count,frontend_latency_cycles=elapsed,forwarded_int8_sha256=row['forwarded_int8_sha256'],real_junit=1))
    return dict(result='INDEPENDENTLY_AUDITED_MATCHED_INT8_STEMS',experiment=contract['experiment'],rows=rows,
        common_binary_sha256=identity['binary_sha256'],contract_sha256=expected,evidence_manifest_sha256=sha256(ev/'SHA256SUMS'),
        full_network_cycles=None,full_network_rtl_pass=False,accuracy=False,board=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','bundle-root','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('New audit output required')
    result=audit(args.root,args.bundle_root);args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
