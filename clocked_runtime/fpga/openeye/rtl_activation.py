"""Opt-in execution of pinned OpenEye ReLU RTL on actual forwarded tensors.

This bridge measures activation-block service cycles with an always-ready
source/sink. Host file I/O, compilation and process launch are separate wall
costs. It does not imply a connected all-RTL network or board latency.
"""
from pathlib import Path
import hashlib
import json
import platform
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
PIN='9a742fbd42a55ebfa8fcd39aa41dfa36a367572f'
PROTOCOL='openeye-relu-rtl-bridge-v1'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def tensor_sha(values):return hashlib.sha256(np.asarray(values,dtype='<i8').tobytes()).hexdigest()


class ReLUBackend:
    top='openeye_relu_file_tb'
    extra_rtl=()
    def __init__(self,openeye,private,evidence,*,width=64,lanes=4,fix_signed=True):
        self.private=Path(private).resolve();self.evidence=Path(evidence).resolve()
        if self.private.exists()or self.evidence.exists()or any(ROOT==p or ROOT in p.parents for p in (self.private,self.evidence)):
            raise ValueError('New private/evidence paths outside Git required')
        if self.private==self.evidence or self.private in self.evidence.parents or self.evidence in self.private.parents:
            raise ValueError('Disjoint activation private/evidence paths required')
        if width not in (8,64)or lanes not in (1,4):raise ValueError('Unsupported activation configuration')
        openeye=Path(openeye)
        if subprocess.check_output(['git','-C',str(openeye),'rev-parse','HEAD'],text=True).strip()!=PIN or subprocess.check_output(['git','-C',str(openeye),'status','--porcelain']):
            raise ValueError('Require clean pinned OpenEye for activation RTL')
        self.private.mkdir(parents=True,mode=0o700);self.evidence.mkdir(parents=True)
        source=subprocess.check_output(['git','-C',str(openeye),'show','HEAD:hdl/af_cluster.v'])
        af=self.private/'af_cluster.v';af.write_bytes(source)
        lane_patch=ROOT/'fpga/openeye/patches/openeye-activation-lane-select.patch'
        subprocess.run(['patch','--batch','--fuzz=0',str(af),str(lane_patch)],check=True,capture_output=True)
        patch=ROOT/'fpga/openeye/patches/openeye-relu-signed.patch'
        if fix_signed:
            # Keep source input and corrected output in the private snapshot.
            (self.private/'upstream-af_cluster.v').write_bytes(source)
            subprocess.run(['patch','--batch','--fuzz=0',str(af),str(patch)],check=True,capture_output=True)
        tb=self.private/(self.top+'.sv');tb.write_bytes((ROOT/'fpga/openeye/rtl'/(self.top+'.sv')).read_bytes())
        self.width,self.lanes=width,lanes;self.binary=self.private/'relu.vvp';self.calls=[]
        extras=[]
        for name in self.extra_rtl:
            path=self.private/name;path.write_bytes((ROOT/'fpga/openeye/rtl'/name).read_bytes());extras.append(path)
        parameters=([f'-P{self.top}.WIDTH={width}',f'-P{self.top}.LANES={lanes}']if not extras else [])
        command=['iverilog','-g2012','-s',self.top,*parameters,'-o',str(self.binary),str(af),str(tb),*map(str,extras)]
        started=time.perf_counter_ns();run=subprocess.run(command,capture_output=True,text=True)
        (self.private/'compile.log').write_text(run.stdout+run.stderr);run.check_returncode()
        self.identity=dict(protocol=PROTOCOL,openeye_pin=PIN,width=width,lanes=lanes,pipeline_depth=1,
            upstream_sha256=hashlib.sha256(source).hexdigest(),rtl_sha256=sha(af),testbench_sha256=sha(tb),
            patch_sha256=sha(patch)if fix_signed else None,binary_sha256=sha(self.binary),
            lane_patch_sha256=sha(lane_patch),
            adapter_rtl_sha256={p.name:sha(p)for p in extras},
            compiler=subprocess.check_output(['iverilog','-V'],stderr=subprocess.DEVNULL,text=True).splitlines()[0],
            host=platform.platform(),compile_wall_ns=time.perf_counter_ns()-started,
            semantics='signed two-complement ReLU; no quantization or saturation')
        (self.evidence/'source-identity.json').write_text(json.dumps(self.identity,indent=2)+'\n')

    def run(self,values,site,*,enabled=True,bubbles=0):
        a=np.asarray(values)
        if a.dtype.kind not in 'iu'or not a.size or a.dtype.kind=='u'and a.max()>(1<<63)-1:
            raise ValueError('Require nonempty signed integer activation')
        low=-(1<<(self.width-1));high=(1<<(self.width-1))-1
        if a.min()<low or a.max()>high or type(enabled)is not bool or type(bubbles)is not int or not 0<=bubbles<=8:
            raise ValueError('Activation input outside configuration')
        if sha(self.binary)!=self.identity['binary_sha256']:raise ValueError('Activation binary changed')
        index=len(self.calls);root=self.private/f'command-{index:04d}';root.mkdir()
        mask=(1<<self.width)-1;digits=self.width//4
        input_file=root/'input.hex';output_file=root/'output.hex'
        input_file.write_text(''.join(f'{int(v)&mask:0{digits}x}\n'for v in a.flat))
        started=time.perf_counter_ns()
        run=subprocess.run(['vvp',str(self.binary),f'+INPUT={input_file}',f'+OUTPUT={output_file}',
            f'+COUNT={a.size}',f'+RELU={int(enabled)}',f'+BUBBLES={bubbles}'],capture_output=True,text=True,timeout=180)
        wall=time.perf_counter_ns()-started;(root/'simulation.log').write_text(run.stdout+run.stderr);run.check_returncode()
        matches=re.findall(r'DOMINO_RELU_RTL_DONE count=(\d+) groups=(\d+) cycles=(\d+) depth=(\d+) bubbles=(\d+)',run.stdout)
        if len(matches)!=1:raise ValueError('Missing activation RTL completion/counters')
        count,groups,cycles,depth,idle=map(int,matches[0])
        if count!=a.size or groups!=(a.size+self.lanes-1)//self.lanes or depth!=1 or idle!=bubbles or cycles!=groups*(1+bubbles):
            raise ValueError('Activation RTL counter mismatch')
        tokens=output_file.read_text().splitlines()
        if len(tokens)!=a.size or any(not re.fullmatch('[0-9a-fA-F]{'+str(digits)+'}',v)for v in tokens):
            raise ValueError('Incomplete/unknown activation output')
        decoded=[int(v,16)for v in tokens]
        actual=np.asarray([v-(1<<self.width)if v&(1<<(self.width-1))else v for v in decoded],dtype=np.int64).reshape(a.shape)
        expected=np.maximum(a,0)if enabled else a
        if not np.array_equal(actual,expected):raise AssertionError('OpenEye ReLU RTL differs from independent signed golden')
        record=dict(index=index,site=site,result='PASS',protocol=PROTOCOL,enabled=enabled,
            input_shape=list(a.shape),checked_values=int(a.size),input_sha256=tensor_sha(a),
            output_sha256=tensor_sha(actual),reference_sha256=tensor_sha(expected),
            input_file_sha256=sha(input_file),output_file_sha256=sha(output_file),
            total_cycles=cycles,active_cycles=groups,injected_idle_cycles=groups*bubbles,
            lanes=self.lanes,pipeline_depth=depth,clock_period_ns=10,clock_scope='simulation clock; not post-route Fmax',
            rtl_service_ns=cycles*10,host_simulator_wall_ns=wall,
            scope='independent activation RTL bridge; reset/file/process/transport excluded',
            network_connected_rtl=False,board=False)
        path=self.evidence/f'activation-{index:04d}.json';path.write_text(json.dumps(record,indent=2)+'\n')
        suite=ET.Element('testsuite',name='OpenEye signed ReLU RTL bridge',tests='1',failures='0',errors='0',skipped='0')
        ET.SubElement(suite,'testcase',name=str(site),classname=PROTOCOL)
        ET.ElementTree(suite).write(self.evidence/f'junit-{index:04d}.xml',encoding='utf-8',xml_declaration=True)
        self.calls.append(record)
        return actual


class PostopsBackend(ReLUBackend):
    """Execute affine/requant/residual+ReLU on actual RTL, not a cycle formula."""
    top='openeye_postops_file_tb'
    extra_rtl=('openeye_postops_stream.sv',)

    def __init__(self,openeye,private,evidence):
        super().__init__(openeye,private,evidence,width=64,lanes=1)
        self.identity.update(protocol='openeye-postops-rtl-bridge-v1',pipeline_depth=4,
            semantics='int64 affine/residual alignment; ties-to-even shift; pinned signed ReLU; saturation [-127,127]')
        (self.evidence/'source-identity.json').write_text(json.dumps(self.identity,indent=2)+'\n')

    def _execute(self,a,b,m,bias,n,shift,residual,relu,site,expected,bubbles=0):
        if type(bubbles)is not int or not 0<=bubbles<=8:raise ValueError('Invalid post-op idle injection')
        if sha(self.binary)!=self.identity['binary_sha256']:raise ValueError('Post-op binary changed')
        index=len(self.calls);root=self.private/f'command-{index:04d}';root.mkdir()
        def expand(value):return np.broadcast_to(value,a.shape).reshape(-1)
        arrays=[expand(v)for v in (a,b,m,bias,n)];mask=(1<<64)-1
        path=root/'input.hex';output=root/'output.hex'
        with path.open('x')as handle:
            for row in zip(*arrays):
                handle.write(' '.join(f'{int(v)&mask:016x}'for v in row)+f' {shift} {int(residual)} {int(relu)}\n')
        started=time.perf_counter_ns()
        run=subprocess.run(['vvp',str(self.binary),f'+INPUT={path}',f'+OUTPUT={output}',f'+COUNT={a.size}',f'+BUBBLES={bubbles}'],capture_output=True,text=True,timeout=300)
        wall=time.perf_counter_ns()-started;(root/'simulation.log').write_text(run.stdout+run.stderr);run.check_returncode()
        matches=re.findall(r'DOMINO_POSTOPS_RTL_DONE count=(\d+) cycles=(\d+) first=(\d+) active=(\d+) idle=(\d+)',run.stdout)
        if len(matches)!=1:raise ValueError('Missing RTL post-op counters')
        count,cycles,first,active,idle=map(int,matches[0])
        if count!=a.size or cycles!=a.size+3+(a.size-1)*bubbles or first!=4 or active!=a.size or active+idle!=cycles:
            raise ValueError('Post-op service counter mismatch')
        tokens=output.read_text().splitlines()
        if len(tokens)!=a.size or any(not re.fullmatch('[0-9a-fA-F]{16}',v)for v in tokens):raise ValueError('Unknown or missing post-op RTL outputs')
        words=[int(v,16)for v in tokens]
        actual=np.asarray([v-(1<<64)if v>>63 else v for v in words],dtype=np.int64).reshape(a.shape)
        if not np.array_equal(actual,expected):raise AssertionError('RTL post-op differs from independent fixed-point golden')
        record=dict(index=index,site=site,result='PASS',protocol=self.identity['protocol'],operation='residual'if residual else'affine',relu=relu,
            checked_values=int(a.size),input_shape=list(a.shape),input_sha256=tensor_sha(a),
            output_sha256=tensor_sha(actual),reference_sha256=tensor_sha(expected),
            private_input_sha256=sha(path),private_output_sha256=sha(output),
            total_cycles=cycles,active_cycles=active,first_output_cycle=first,drain_cycles=3,
            injected_idle_cycles=(a.size-1)*bubbles,pipeline_depth=4,initiation_interval=1,
            clock_period_ns=10,clock_scope='simulation clock; not validated Fmax',host_simulator_wall_ns=wall,
            overlapping_stages=['affine/alignment','ties-to-even requantization','OpenEye signed ReLU/bypass','saturation'],
            scope='standalone RTL service interval; host transfers and reset excluded',network_connected_rtl=False,board=False)
        (self.evidence/f'postops-{index:04d}.json').write_text(json.dumps(record,indent=2)+'\n')
        suite=ET.Element('testsuite',name='OpenEye fixed-point post-op RTL bridge',tests='1',failures='0',errors='0',skipped='0')
        ET.SubElement(suite,'testcase',name=str(site),classname=self.identity['protocol'])
        ET.ElementTree(suite).write(self.evidence/f'junit-{index:04d}.xml',encoding='utf-8',xml_declaration=True)
        self.calls.append(record);return actual

    def affine(self,values,multiplier,bias,shift=24,relu=False,timings=None,site='affine',bubbles=0):
        from fpga.openeye.runtime_activations import affine_fixed
        from fpga.openeye.experiments.host_postops_fused import affine_inputs
        a,m,b=affine_inputs(values,multiplier,bias,shift)
        expected=affine_fixed(a,m,b,shift,relu=relu)
        return self._execute(a,0,m[:,None,None],b[:,None,None],0,shift,False,relu,site,expected,bubbles)

    def residual(self,main,main_scale,skip,skip_scale,output_scale,shift=24,timings=None,site='residual',bubbles=0):
        from fpga.openeye.runtime_activations import residual_add
        from fpga.openeye.experiments.host_postops_fused import residual_inputs
        a,b,c=residual_inputs(main,main_scale,skip,skip_scale,output_scale,shift)
        expected=residual_add(a,main_scale,b,skip_scale,output_scale,shift)
        return self._execute(a,b,c[0],0,c[1],shift,True,True,site,expected,bubbles)


class GAPBackend(ReLUBackend):
    top='openeye_gap_file_tb'
    extra_rtl=('openeye_gap_stream.sv',)

    def __init__(self,openeye,private,evidence):
        super().__init__(openeye,private,evidence,width=64,lanes=1)
        self.identity.update(protocol='openeye-gap-rtl-bridge-v1',pipeline_depth=1,
            semantics='nonnegative INT8 spatial sum/divide, ties-to-even')
        (self.evidence/'source-identity.json').write_text(json.dumps(self.identity,indent=2)+'\n')

    def run(self,values,site='gap',bubbles=0):
        from fpga.openeye.runtime_activations import global_average_pool,integer_array
        a=integer_array(values,low=0)
        if a.ndim!=3 or not 1<=a.shape[1]*a.shape[2]<=65535 or type(bubbles)is not int or not 0<=bubbles<=8:
            raise ValueError('Unsupported GAP geometry')
        expected=global_average_pool(a)
        if sha(self.binary)!=self.identity['binary_sha256']:raise ValueError('GAP binary changed')
        index=len(self.calls);root=self.private/f'command-{index:04d}';root.mkdir()
        path=root/'input.hex';output=root/'output.hex';path.write_text(''.join(f'{int(v):02x}\n'for v in a.flat))
        started=time.perf_counter_ns()
        run=subprocess.run(['vvp',str(self.binary),f'+INPUT={path}',f'+OUTPUT={output}',f'+CHANNELS={a.shape[0]}',
            f'+PIXELS={a.shape[1]*a.shape[2]}',f'+BUBBLES={bubbles}'],capture_output=True,text=True,timeout=300)
        wall=time.perf_counter_ns()-started;(root/'simulation.log').write_text(run.stdout+run.stderr);run.check_returncode()
        matches=re.findall(r'DOMINO_GAP_RTL_DONE inputs=(\d+) outputs=(\d+) cycles=(\d+) active=(\d+)',run.stdout)
        if len(matches)!=1:raise ValueError('Missing GAP completion/counters')
        inputs,outputs,cycles,active=map(int,matches[0])
        if inputs!=a.size or outputs!=a.shape[0]or cycles!=a.size+(a.size-1)*bubbles or active!=a.size:
            raise ValueError('GAP cycle/coverage mismatch')
        tokens=output.read_text().splitlines()
        if len(tokens)!=outputs or any(not re.fullmatch('[0-9a-fA-F]{16}',v)for v in tokens):raise ValueError('Missing/unknown GAP output')
        actual=np.asarray([int(v,16)for v in tokens],dtype=np.int64)
        if not np.array_equal(actual,expected):raise AssertionError('GAP RTL differs from independent golden')
        record=dict(index=index,site=site,result='PASS',protocol=self.identity['protocol'],input_shape=list(a.shape),
            checked_values=outputs,input_values=inputs,input_sha256=tensor_sha(a),output_sha256=tensor_sha(actual),reference_sha256=tensor_sha(expected),
            total_cycles=cycles,active_cycles=active,injected_idle_cycles=(a.size-1)*bubbles,
            first_output_cycle=a.shape[1]*a.shape[2]+(a.shape[1]*a.shape[2]-1)*bubbles,
            clock_period_ns=10,clock_scope='simulation clock; divider timing unverified',host_simulator_wall_ns=wall,
            scope='standalone GAP RTL service; host transfers/reset excluded',network_connected_rtl=False,board=False)
        (self.evidence/f'gap-{index:04d}.json').write_text(json.dumps(record,indent=2)+'\n')
        suite=ET.Element('testsuite',name='GAP RTL bridge',tests='1',failures='0',errors='0',skipped='0')
        ET.SubElement(suite,'testcase',name=str(site),classname=self.identity['protocol'])
        ET.ElementTree(suite).write(self.evidence/f'junit-{index:04d}.xml',encoding='utf-8',xml_declaration=True)
        self.calls.append(record);return actual
