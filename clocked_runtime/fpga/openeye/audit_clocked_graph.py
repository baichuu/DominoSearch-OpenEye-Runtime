#!/usr/bin/env python3
"""Independent image→logit RTL audit; publishes aggregate evidence only.

All actual raw DMA slots, accepted words and forwarded intermediate values are
checked. Goldens execute solely in this post-execution checker. They never
initialize the running graph. Missing evidence yields null network cycles.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET
import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from fpga.openeye.validate_graph_program import validate,PIN,read_words,require,sha
from fpga.openeye.prepare_packed_layer_fixture import packed_layer,decode_and_quantize,conv_iohw,read_float32_le,verify_source_file
from fpga.openeye.run_residual_block import batchnorm,bn_coefficients,positive_scale
from fpga.openeye.audit_fixedpoint_frontends import independent_stem
from fpga.openeye.rtl_stem import signed_convolution
from fpga.openeye.rtl_logits import coefficients as logit_coefficients
from fpga.openeye.runtime_activations import convolution,affine_fixed,residual_add,global_average_pool
from fpga.openeye.graph_layout import windows,map_input
from fpga.openeye.graph_mapper import GraphMapper


class WordStream:
    def __init__(self,path,digits=16):self.file=path.open();self.digits=digits;self.count=0
    def take(self,count):
        result=[]
        for _ in range(count):
            token=self.file.readline().strip()
            require(bool(re.fullmatch('[0-9a-fA-F]{'+str(self.digits)+'}',token)),'Missing/unknown actual stream word')
            result.append(int(token,16));self.count+=1
        return result
    def finish(self):require(not self.file.read().strip(),'Unexpected trailing actual stream values');self.file.close()


def signed(values,bits=64):
    return np.asarray([v-(1<<bits)if v>>(bits-1)else v for v in values],dtype=np.int64)


def parse_timeline(log,instructions,stop_after=None):
    require(not re.search(r'(?:%Error|%Fatal|FATAL:|Assertion failed)',log),'Actual RTL reported failure')
    start=re.findall(r'GRAPH_INSTRUCTION_START index=(\d+) op=(\d+) cycles=(\d+)',log)
    end=re.findall(r'GRAPH_INSTRUCTION_DONE index=(\d+) cycles=(\d+) inputs=(\d+) raw_words=(\d+)',log)
    count=len(instructions)-1 if stop_after is None else stop_after+1
    require(0<=count<len(instructions),'Invalid audited prefix boundary')
    starts_seen=[int(i)for i,_,_ in start]
    allowed=[list(range(len(instructions)))]if stop_after is None else[list(range(count)),list(range(count+1))]
    require(starts_seen in allowed,'Missing/reordered instruction start markers')
    require([int(i)for i,_,_,_ in end]==list(range(count)),'Missing/reordered instruction completion markers')
    starts={int(i):(int(op),int(c))for i,op,c in start};ends={int(i):(int(c),int(a),int(r))for i,c,a,r in end}
    for i in starts:
        row=instructions[i]
        require(starts[i][0]==row[0],'Executed opcode differs from frozen program')
        if i<count:require(ends[i][0]>starts[i][1]and(i+1 not in starts or starts[i+1][1]==ends[i][0]),'Gap/overlap in controller timeline')
    input_total=raw_total=0
    for i,row in enumerate(instructions[:count]):
        if row[0]==1:input_total+=row[2];raw_total+=row[6]
        require(ends[i][1:]==(input_total,raw_total),'Missing/extra per-instruction transfer counts')
    states={};intervals={}
    for i,state,cycles,words in re.findall(r'GRAPH_CORE_STATE index=(\d+) state=(\d+) cycles=(\d+) input_words=(\d+)',log):
        i,state,cycles,words=map(int,(i,state,cycles,words))
        require((i,state)not in states and 0<=i<count and instructions[i][0]==1 and
                0<=state<64 and 0<=words<=cycles,'Invalid/duplicate native core counter')
        states[i,state]=(cycles,words)
    for i,cycles,retries in re.findall(r'GRAPH_CORE_INTERVAL index=(\d+) cycles=(\d+) retry_cycles=(\d+)',log):
        i,cycles,retries=map(int,(i,cycles,retries));require(i not in intervals,'Duplicate native core interval')
        rows=[v for(k,_),v in states.items()if k==i]
        require(sum(v[0]for v in rows)==cycles and 0<=retries<=cycles,'Native state counters do not sum to observed interval')
        expected_inputs=ends[i][1]-(ends[i-1][1]if i else 0)
        require(sum(v[1]for v in rows)==expected_inputs==instructions[i][2],'Native accepted-word counter mismatch')
        intervals[i]=(cycles,retries)
    require(set(intervals)=={i for i,row in enumerate(instructions[:count])if row[0]==1},'Missing native core interval counters')
    return starts,ends,states,intervals


def audit(args):
    stop_after=getattr(args,'stop_after_instruction',None)
    root=args.root.resolve();structural=validate(root);prep=json.loads((root/'preparation.json').read_text())
    execution=json.loads((root/'execution.json').read_text());identity=json.loads((root/'clocked-build/identity.json').read_text())
    require(execution['returncode']==0 and execution['identity_sha256']==sha(root/'clocked-build/identity.json'),'Failed/foreign graph execution')
    require(identity['openeye_pin']==PIN and identity['preparation_sha256']==sha(root/'preparation.json')and
            identity['parameters']['USE_STEM']==1 and sha(root/'clocked-build/obj/Vopeneye_graph_tb')==identity['binary_sha256'],
            'Wrong actual complete-graph binary identity')
    for name,digest in identity['sources'].items():
        require(not Path(name).is_absolute()and '..'not in Path(name).parts and sha(root/name)==digest,'Changed actual HDL/header input')
    for name,digest in execution['files'].items():require(sha(root/name)==digest,'Changed actual graph execution trace')
    if identity.get('driver_protocol')=='verilated-saveable-low-edge-v1':
        from fpga.openeye.clocked_resume import validate_chain
        require(execution['resume_chain_sha256']==sha(root/'resume-chain.json'),'Changed resume execution chain')
        resume_chain=validate_chain(root,complete=stop_after is None)
    program=[[int(line[i:i+8],16)for i in range(0,len(line),8)][::-1]for line in(root/'program.hex').read_text().splitlines()]
    log=(root/'simulation.log').read_text();starts,ends,states,intervals=parse_timeline(log,program,stop_after)
    packed=args.bundle_root/prep['model']/'packed-v1';manifest=json.loads((packed/'manifest.json').read_text())
    contract=json.loads((root/'arithmetic-contract.json').read_text());unsigned=dict(contract);expected_contract=unsigned.pop('contract_sha256')
    require(hashlib.sha256(json.dumps(unsigned,sort_keys=True,separators=(',',':')).encode()).hexdigest()==expected_contract and
            contract['experiment']=='whole-network-int8-fixedpoint-v1','Changed arithmetic/input contract')
    vectors=args.bundle_root/'mac20/test-vectors-100';v=json.loads((vectors/'manifest.json').read_text());path=vectors/v['input']['file']
    require(path.parent==vectors and sha(path)==v['input']['sha256']==contract['vector_sha256']and
            path.stat().st_size==100*3*224*224*4,'Changed original verification vector')
    sample=contract['sample_index'];require(type(sample)is int and 0<=sample<100,'Invalid verification-vector index')
    image=np.memmap(path,dtype='<f4',mode='r',shape=(100,3,224,224))[sample].copy().astype(np.float64).transpose(0,2,1)
    input_scale=positive_scale(image);image=np.clip(np.rint(image/input_scale),-127,127).astype(np.int64)
    require(input_scale==contract['input_scale']and hashlib.sha256(image.astype('i1').tobytes()).hexdigest()==contract['quantized_input_sha256'],
            'Changed signed image encoding')
    require(np.array_equal(signed(read_words(root/'initial-memory.hex',16)).reshape(image.shape),image),'Initial SRAM contains an intermediate or different image')
    cache={}
    def weights(layer):
        if layer not in cache:
            item=packed_layer(manifest,layer);values,q,_=decode_and_quantize(packed,item)
            w=(np.asarray(values,dtype=np.int64).reshape(item['source_weight_shape'])if layer=='fc'else
               np.asarray(conv_iohw(values,item['source_weight_shape'],item['grouping']['grouped_order']),dtype=np.int64))
            cache[layer]=(w,q)
        return cache[layer]
    stem_w,stem_q=weights('conv1');alpha,beta,_=batchnorm(packed,manifest,'bn1')
    m,b,scale=bn_coefficients(signed_convolution(image,stem_w),input_scale,stem_q['scale'],alpha,beta,True)
    stem_words=read_words(root/'stem-weights.hex',2)
    require(np.array_equal(signed(stem_words,8).reshape(stem_w.shape),stem_w),'Stem weights differ from immutable source')
    expected_coeff=''.join(f'{int(x)&((1<<64)-1):016x} {int(y)&((1<<64)-1):016x}\n'for x,y in zip(m,b))
    require((root/'stem-coefficients.hex').read_text()==expected_coeff,'Stem BN coefficients differ from source/calibration')
    stem_golden=independent_stem(image,stem_w,m,b)
    actual_stem=np.asarray(read_words(root/'actual-stem.hex',2),dtype=np.int64).reshape(64,56,56)
    require(np.array_equal(actual_stem,stem_golden),'Actual connected stem differs from independent shift/add oracle')
    stem_marker=re.findall(r'GRAPH_STEM_DONE values=(\d+) cycles_from_image=(\d+)',log)
    require(len(stem_marker)==1 and int(stem_marker[0][0])==200704,'Missing actual stem timeline/boundary')
    stem_cycles=int(stem_marker[0][1])
    component_markers=re.findall(r'GRAPH_STEM_COMPONENT name=(\w+) values=(\d+) first=(\d+) last=(\d+)',log)
    require([row[0]for row in component_markers]==['conv7x7','bn_requant_relu','maxpool'],'Missing stem function intervals')
    stem_components=[]
    for name,count,first,last in component_markers:
        count,first,last=map(int,(count,first,last))
        require(count==(200704 if name=='maxpool'else 802816)and 0<first<=last,'Incomplete actual stem function values')
        stem_components.append(dict(operation=name,accepted_values=count,first_output_edge=first,last_output_edge=last,
            output_span_cycles=last-first,scope='overlapping pipeline output span; do not sum with other stem components'))
    memory=np.zeros(prep['memory_values'],dtype=np.int64);memory[:200704]=actual_stem.reshape(-1)
    scales={0:scale};shapes={0:(64,56,56)}
    coefficients=[]
    for line in(root/'coefficients.hex').read_text().splitlines():coefficients.append(signed([int(v,16)for v in line.split()]).tolist())
    maps=read_words(root/'maps.hex',8);static=WordStream(root/'packets.hex');accepted=WordStream(root/'actual-accepted.hex');raw=WordStream(root/'actual-raw.hex')
    forwarded=(root/'actual-forwarded.hex').open();forwarded_count=0;stage_records=[];post_records=[]
    mapper=GraphMapper(root/'generated',prep['optimized']);geometry={};stages={s['first_instruction']:s for s in prep['stages']};stage_by_post={s['last_instruction']:s for s in prep['stages']}
    pending={};reference_outputs={};next_ip=0;last_stage=None
    def forwarded_output(ip,destination,expected):
        nonlocal forwarded_count
        expected=np.asarray(expected,dtype=np.int64).reshape(-1);actual=[]
        for offset,value in enumerate(expected):
            tokens=forwarded.readline().split()
            require(len(tokens)==3 and all(re.fullmatch('[0-9a-fA-F]{'+str(n)+'}',t)for t,n in zip(tokens,(8,8,16))),
                    'Missing/malformed actual intermediate forwarding value')
            instruction,address,word=map(lambda v:int(v,16),tokens)
            require(instruction==ip and address==destination+offset,'Wrong actual intermediate order/address')
            decoded=word-(1<<64)if word>>63 else word
            require(decoded==int(value),'Actual forwarded tensor differs from independent source golden')
            actual.append(decoded);forwarded_count+=1
        result=np.asarray(actual,dtype=np.int64);memory[destination:destination+result.size]=result;return result
    executed=program[:-1]if stop_after is None else program[:stop_after+1]
    for ip,row in enumerate(executed):
        op=row[0]
        if op==1:
            if ip in stages:
                last_stage=stages[ip];name=last_stage['layer'];w,q=weights(name)
                first=row;shape=(512,)if name=='fc'else(row[12],row[10],row[11])
                a=memory[row[9]:row[9]+int(np.prod(shape))].reshape(shape).copy()
                require(tuple(shapes.get(row[9],()))==shape,'Convolution input is not the actual preceding tensor')
                stride=1 if name=='fc'or name.endswith('conv2')or '.1.'in name or name.startswith('layer1.')else 2
                reference_outputs[name]=(a@w.T if name=='fc'else convolution(a,w,stride))
                pending=dict(name=name,source=a,weights=w,quant=q,stride=stride,input_scale=scales[row[9]],command=0)
            stage=last_stage;name=pending['name'];entry=stage['commands'][pending['command']]
            require(entry['instruction']==ip,'Wrong source stage command attribution');pending['command']+=1
            c=entry['channel_start'];f=entry['filter_start'];linear=name=='fc'
            if linear:
                a=pending['source'][c:c+64];w=pending['weights'][f:f+8,c:c+64];tw=th=1;s=1;k=1;nf=8
                coords=[(0,0,v)for v in(0,4,1,5,2,6,3,7)];expected_raw=[int(v)&0xffffffff for v in(w@a)[[0,4,1,5,2,6,3,7]]]
                expected_map=[v for _,_,local_filter in coords for v in(f+local_filter,0xffffffff)]
            else:
                sw,sh=pending['source'].shape[1:];window=windows(sw,sh,pending['stride'])[entry['window']]
                a=map_input(pending['source'][c:c+64],window);nf=16 if prep['optimized']else 8
                w=pending['weights'][c:c+64,f:f+nf]
                if w.shape[-2:]==(1,1):
                    embedded=np.zeros((*w.shape[:2],3,3),dtype=np.int64);embedded[:,:,1,1]=w[:,:,0,0];w=embedded
                tw,th=a.shape[1:];s=window['execution_stride'];k=w.shape[2];key=(tw,th,nf,k,s)
                if key not in geometry:
                    p=mapper.parameters(tw,th,nf,k,s);zero=np.zeros((nf,p.psum_size_x,p.psum_size_y),dtype=np.int64)
                    geometry[key]=mapper.coordinates(mapper.hardware,p,zero,io.StringIO())
                coords=geometry[key];valid=[(x,y,f_)for x,y,f_ in coords if 0<=f_<nf]
                ew=max(tw,(max(x for x,_,_ in valid)+1)*s);eh=max(th,(max(y for _,y,_ in valid)+1)*s)
                ref=convolution(np.pad(a,((0,0),(0,ew-tw),(0,eh-th))),w,s)
                lanes=[int(ref[f_,x,y])if 0<=f_<nf and 0<=x<ref.shape[1]and 0<=y<ref.shape[2]else 0 for x,y,f_ in coords]
                expected_raw=[(lanes[i]&0xffffffff)|((lanes[i+1]&0xffffffff)<<32)for i in range(0,len(lanes),2)]
                expected_map=[];logical=window['logical_output']
                for x,y,local_filter in coords:
                    sx=x-window['crop_start']+window['output_start']if window['kind']=='halo'else x
                    good=(0<=local_filter<nf and 0<=y<logical[1]and window['output_start']<=sx<window['output_stop'])
                    expected_map.append((f+local_filter)*logical[0]*logical[1]+sx*logical[1]+y if good else 0xffffffff)
            require(maps[row[5]:row[5]+2*row[6]]==expected_map,'Actual scatter map differs from native coordinate and source geometry')
            require(len(expected_raw)==row[6]and raw.take(row[6])==expected_raw,'Raw OpenEye result differs, including padded slots')
            packet=static.take(row[2]);iact=row[4]
            if iact:
                if linear:values=a.reshape(-1).tolist()
                else:values=[int(a[g+offset,x,y])for g in range(0,64,row[21])for y in range(th)for x in range(tw)for offset in range(row[21])]
                words=[sum(int(v)<<(8*j)for j,v in enumerate(values[i:i+8]))for i in range(0,len(values),8)]
                live=[v for i in range(0,len(words),8)for v in words[i:i+8][::-1]]
                if linear:live+=[0]*(iact-len(live))
                require(len(live)==iact and all(v==0 for v in packet[row[3]:row[3]+iact]),'Static model ROM contains unexpected live activation payload')
                packet[row[3]:row[3]+iact]=live
            require(accepted.take(row[2])==packet,'Actual accepted packet differs from model setup/live SRAM activation')
        elif op in(2,5):
            stage=stage_by_post[ip];name=stage['layer'];reference=reference_outputs[name]
            if op==2:
                prefix=name.rsplit('.',1)[0]+('.bn1'if name.endswith('conv1')else'.bn2')
                if name.endswith('downsample.0'):prefix=name[:-1]+'1'
                alpha,beta,_=batchnorm(packed,manifest,prefix)
                m,b,scale=bn_coefficients(reference,pending['input_scale'],pending['quant']['scale'],alpha,beta,bool(row[6]))
                require(coefficients[row[4]:row[4]+len(m)]==list(map(list,zip(m.tolist(),b.tolist()))),'BN calibration/coefficients differ from immutable source')
                golden=affine_fixed(reference,m,b,24,relu=bool(row[6]));shapes[row[2]]=golden.shape;scales[row[2]]=scale
                actual=forwarded_output(ip,row[2],golden)
            else:
                bias_record=next(v for v in manifest['auxiliary_tensors']if v['state_key']=='fc.bias')
                bias=read_float32_le(verify_source_file(packed,bias_record,'float32'))
                m,b=logit_coefficients(pending['input_scale']*pending['quant']['scale'],bias)
                require(coefficients[row[4]:row[4]+1000]==[[m,int(v)]for v in b],'FC bias/Q40 coefficients differ from immutable source')
                golden=np.asarray([int(v)*m+int(b_)for v,b_ in zip(reference,b)],dtype=np.int64)
                actual=forwarded_output(ip,row[2],golden);shapes[row[2]]=golden.shape
            start=starts[stage['first_instruction']][1];finish=ends[ip][0]-1
            stage_records.append(dict(layer=name,scheme=str(packed_layer(manifest,name)['n'])+':'+str(packed_layer(manifest,name)['m']),
                commands=len(stage['commands']),checked_values=int(actual.size),latency_cycles=finish-start,
                convolution_dma_accumulation_cycles=sum(ends[c['instruction']][0]-starts[c['instruction']][1]for c in stage['commands']),
                postops_cycles=ends[ip][0]-1-starts[ip][1],
                output_sha256=hashlib.sha256(actual.astype('<i8').tobytes()).hexdigest()))
            post_records.append(dict(operation='fc_scale_bias_q40'if op==5 else'bn_requant_relu'if row[6]else'bn_requant',
                site=name,checked_values=int(actual.size),latency_cycles=ends[ip][0]-1-starts[ip][1]))
        elif op==3:
            shape=shapes[row[1]];main=memory[row[1]:row[1]+row[3]].reshape(shape);skip=memory[row[6]:row[6]+row[3]].reshape(shape)
            main_scale=scales[row[1]];skip_scale=scales[row[6]];scale=positive_scale(np.maximum(main*main_scale+skip*skip_scale,0))
            m=int(round(main_scale/scale*(1<<24)));b=int(round(skip_scale/scale*(1<<24)))
            require(coefficients[row[4]]==[m,b],'Residual alignment differs from same-source calibration')
            golden=residual_add(main,main_scale,skip,skip_scale,scale)
            actual=forwarded_output(ip,row[2],golden);shapes[row[2]]=shape;scales[row[2]]=scale
            post_records.append(dict(operation='residual_add_requant_relu',site=str(len(post_records)),checked_values=int(actual.size),latency_cycles=ends[ip][0]-1-starts[ip][1]))
        elif op==4:
            a=memory[row[1]:row[1]+row[3]].reshape(shapes[row[1]])
            golden=global_average_pool(a);actual=forwarded_output(ip,row[2],golden)
            shapes[row[2]]=golden.shape;scales[row[2]]=scales[row[1]]
            post_records.append(dict(operation='gap',site='gap',checked_values=int(actual.size),latency_cycles=ends[ip][0]-1-starts[ip][1]))
    if stop_after is None:static.finish()
    else:static.file.close()
    accepted.finish();raw.finish();require(not forwarded.read().strip(),'Unexpected extra forwarded tensor values');forwarded.close()
    if stop_after is not None:
        require(not(root/'actual-output.hex').read_text().strip()and 'GRAPH_OUTPUT_DONE'not in log,'Final output cannot be relabeled as prefix')
        expected_stages=sum(s['last_instruction']<=stop_after for s in prep['stages'])
        require(len(stage_records)==expected_stages and len(post_records)==sum(row[0]in(2,3,4,5)for row in executed),'Incomplete checked prefix operators')
        require(identity.get('driver_protocol')=='verilated-saveable-low-edge-v1'and
                resume_chain['chunks'][-1]['end_status']['last_logged_instruction']==(0xffffffff if stop_after==-1 else stop_after),
                'Snapshot does not end at the checked prefix instruction')
        return dict(result='PASS_CHECKED_CLOCKED_PREFIX_NOT_NETWORK',model=prep['model'],optimized=prep['optimized'],openeye_pin=PIN,
            stop_after_instruction=stop_after,checked_stem_values=200704,checked_forwarded_values=forwarded_count,
            checked_stages=len(stage_records),checked_postops=len(post_records),checked_raw_words=raw.count,accepted_input_words=accepted.count,
            observed_prefix_cycles=resume_chain['chunks'][-1]['end_status']['observed_cycles'],
            preparation_sha256=sha(root/'preparation.json'),binary_sha256=identity['binary_sha256'],
            resume_chain_sha256=sha(root/'resume-chain.json'),full_network_cycles=None,full_network_rtl_pass=False,real_rtl_junit=1)
    require(len(stage_records)==20 and len(post_records)==29,'Incomplete stage/post-op audit')
    output=signed(read_words(root/'actual-output.hex',16));require(output.size==1000 and np.array_equal(output,memory[prep['output_base']:prep['output_base']+1000]),'Final output file differs from actual forwarded logits')
    marker=re.findall(r'GRAPH_OUTPUT_DONE values=(\d+) latency=(\d+) slots=(\d+) engine_cycles=(\d+)',log)
    require(len(marker)==1,'Missing/duplicate complete-network output marker');count,cycles,slots,engine_at_output=map(int,marker[0])
    require(count==1000 and slots==cycles+1 and cycles==stem_cycles+1+engine_at_output,'Network timeline differs from actual stem/controller edge counts')
    end_marker=re.findall(r'GRAPH_PROGRAM_DONE instructions=(\d+) inputs=(\d+) raw_words=(\d+) cycles=(\d+)',log)
    require(len(end_marker)==1 and tuple(map(int,end_marker[0][:3]))==(len(program)-1,accepted.count,raw.count),'Incomplete final program/transfer counters')
    return dict(result='PASS_INDEPENDENT_COMPLETE_CLOCKED_NETWORK_AUDIT',experiment=prep['experiment'],model=prep['model'],optimized=prep['optimized'],
        openeye_pin=PIN,source_checkpoint_sha256=manifest['checkpoint']['sha256'],source_scheme_sha256=manifest['scheme']['sha256'],
        arithmetic_contract_sha256=expected_contract,input_sha256=contract['quantized_input_sha256'],
        preparation_sha256=sha(root/'preparation.json'),binary_sha256=identity['binary_sha256'],
        checked_stem_values=200704,checked_forwarded_values=forwarded_count,checked_logits=1000,
        resume_chain_sha256=sha(root/'resume-chain.json')if identity.get('driver_protocol')=='verilated-saveable-low-edge-v1'else None,
        resumed_chunks=len(resume_chain['chunks'])if identity.get('driver_protocol')=='verilated-saveable-low-edge-v1'else 1,
        stem_components=stem_components,
        accepted_input_words=accepted.count,checked_raw_words=raw.count,stages=stage_records,postops=post_records,
        stem_latency_cycles=stem_cycles,full_network_cycles=cycles,observation_slots=slots,full_network_rtl_pass=True,
        output_sha256=hashlib.sha256(output.astype('<i8').tobytes()).hexdigest(),real_rtl_junit=1,
        native_core_states=[dict(instruction=i,state=state,cycles=c,input_words=w)for(i,state),(c,w)in states.items()],
        clock_period_ns=10,clock_scope='simulation clock only; no Fmax validation',
        interval='first signed image byte accepted to last Q40 logit stored; one clock, no reset across graph',
        activation_measurement='BN, requantization and ReLU execute in one fused RTL pipeline; reported together, not separately inferred',
        model_initialization='before interval; all inference DMA, SRAM, PSUM, postops and stalls included',
        hardware_scope='common scalar stem + shared OpenEye core + clocked graph scratch and controller',
        accuracy=False,synthesis=False,post_route=False,board=False,host=identity['host'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in('root','bundle-root','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--stop-after-instruction',type=int,help='Check a completed source prefix, never certify full network')
    args=parser.parse_args();require(not args.output.exists(),'New public evidence directory required');args.output.mkdir()
    try:
        report=audit(args)
    except Exception as error:
        # No exception body/tensors are exported from a private evidence audit.
        report=dict(result='FAIL_CLOCKED_NETWORK_AUDIT',failure_type=type(error).__name__,full_network_cycles=None,full_network_rtl_pass=False)
        suite=ET.Element('testsuite',name='Actual clocked network audit',tests='1',failures='1',errors='0',skipped='0')
        case=ET.SubElement(suite,'testcase',name='image-to-q40-logits');ET.SubElement(case,'failure',message='Independent actual RTL evidence audit failed')
        ET.ElementTree(suite).write(args.output/'junit.xml',encoding='utf-8',xml_declaration=True)
        (args.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n');raise
    (args.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    for key,name in(()if args.stop_after_instruction is not None else(('stages','layers.csv'),('postops','postops.csv'),('native_core_states','native-core-states.csv'))):
        rows=report[key]
        with(args.output/name).open('w',newline='')as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    prefix=args.stop_after_instruction is not None
    suite=ET.Element('testsuite',name='Actual checked clocked prefix'if prefix else'Actual complete clocked network',tests='1',failures='0',errors='0',skipped='0')
    ET.SubElement(suite,'testcase',name='prefix-through-'+str(args.stop_after_instruction)if prefix else'image-to-q40-logits',classname='whole-network-int8-fixedpoint-v1')
    ET.ElementTree(suite).write(args.output/'junit.xml',encoding='utf-8',xml_declaration=True)
    print(json.dumps({k:v for k,v in report.items()if k not in('stages','postops','native_core_states')},indent=2))
