"""Actual connected INT8 stem experiment, separate from the FP64 hybrid stem.

No model accuracy or complete-network claim. Weights/coefficients are model
setup; image loading and all subsequent cycles share one observed RTL clock.
"""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
import xml.etree.ElementTree as ET

import numpy as np

from fpga.openeye.rtl_activation import ReLUBackend,PIN,sha,tensor_sha
from fpga.openeye.runtime_activations import integer_array,affine_fixed
from fpga.openeye.experiments.host_postops_fused import affine_inputs

RTL=Path(__file__).with_name('rtl')
FILES=('openeye_stem_conv_stream.sv','openeye_stem_frontend.sv','openeye_stem_frontend_tb.sv',
       'openeye_postops_stream.sv','openeye_postops_elastic.sv','openeye_maxpool3_stream.sv',
       'openeye_network_cycle_monitor.sv')


def signed_convolution(a,w,stride=2):
    a=integer_array(a);w=integer_array(w)
    if a.ndim!=3 or w.ndim!=4 or a.shape[0]!=w.shape[0] or w.shape[2]!=w.shape[3] or w.shape[2]not in(1,3,7) or stride not in(1,2):
        raise ValueError('Unsupported signed stem geometry')
    k=w.shape[2];padded=np.pad(a,((0,0),(k//2,k//2),(k//2,k//2)))
    windows=np.lib.stride_tricks.sliding_window_view(padded,(k,k),axis=(1,2))[:,::stride,::stride]
    result=np.einsum('cxyij,cfji->fxy',windows,w,dtype=np.int64,optimize=True)
    if result.min()<-(1<<31) or result.max()>=(1<<31):raise ValueError('Stem INT32 overflow')
    return result


def maxpool3(a):
    a=integer_array(a,low=0)
    if a.ndim!=3:raise ValueError('Expected CXY pool tensor')
    padded=np.pad(a,((0,0),(1,1),(1,1)))
    return np.lib.stride_tricks.sliding_window_view(padded,(3,3),axis=(1,2))[:,::2,::2].max(axis=(-1,-2))


class StemBackend:
    def __init__(self,openeye,private,evidence,*,simulator='icarus'):
        self.private=Path(private);self.evidence=Path(evidence)
        if self.private.exists() or self.evidence.exists():raise ValueError('New private/evidence roots required')
        if simulator not in('icarus','verilator'):raise ValueError('Unknown simulator')
        self.private.mkdir(parents=True);self.evidence.mkdir(parents=True)
        # Reuse the verified upstream-source/patch preparation, not its outputs.
        self.relu=ReLUBackend(openeye,self.private/'relu-source',self.evidence/'relu-source')
        self.simulator=simulator;self.identity=None;self.calls=[]

    def _build(self,geometry):
        if self.identity is not None:
            if self.identity['geometry']!=geometry or sha(self.binary)!=self.identity['binary_sha256']:
                raise ValueError('Changed stem geometry/binary after build')
            return
        build=self.private/'build';build.mkdir()
        sources=[self.relu.private/'af_cluster.v']
        for name in FILES:
            target=build/name;target.write_bytes((RTL/name).read_bytes());sources.append(target)
        params={**geometry,'SHIFT':geometry['SHIFT']}
        if self.simulator=='icarus':
            self.binary=build/'stem.vvp'
            command=['iverilog','-g2012','-s','openeye_stem_frontend_tb',
                *[f'-Popeneye_stem_frontend_tb.{k}={v}'for k,v in params.items()],'-o',str(self.binary),*map(str,sources)]
        else:
            self.binary=build/'obj/Vopeneye_stem_frontend_tb'
            command=['verilator','--binary','--timing','--top-module','openeye_stem_frontend_tb',
                '--Mdir',str(build/'obj'),'-j','2','-Wno-fatal',
                *[f'-G{k}={v}'for k,v in params.items()],*map(str,sources)]
        started=time.perf_counter_ns();result=subprocess.run(command,capture_output=True,text=True)
        (build/'compile.log').write_text(result.stdout+result.stderr);result.check_returncode()
        self.identity=dict(protocol='openeye-int8-stem-frontend-v1',openeye_pin=PIN,
            geometry=geometry,simulator=self.simulator,binary_sha256=sha(self.binary),
            rtl_sha256={p.name:sha(p)for p in sources},compile_wall_ns=time.perf_counter_ns()-started,
            architecture='common scalar signed INT8 stem + fixed affine/ReLU + streaming maxpool',
            synthesis=False,board=False)
        (self.evidence/'source-identity.json').write_text(json.dumps(self.identity,indent=2)+'\n')

    def run(self,activations,weights,multiplier,bias,shift=24,*,stride=2,sink_stalls=False,site='stem'):
        a=integer_array(activations);w=integer_array(weights)
        reference=signed_convolution(a,w,stride)
        _,m,b=affine_inputs(reference,multiplier,bias,shift)
        expected=maxpool3(affine_fixed(reference,m,b,shift,relu=True))
        geometry=dict(WIDTH=a.shape[1],HEIGHT=a.shape[2],CHANNELS=a.shape[0],FILTERS=w.shape[1],
            KERNEL=w.shape[2],STRIDE=stride,SHIFT=shift)
        self._build(geometry)
        index=len(self.calls);root=self.private/f'command-{index:04d}';root.mkdir()
        image=root/'image.hex';weight=root/'weights.hex';coeff=root/'coefficients.hex';output=root/'output.hex'
        image.write_text(''.join(f'{int(v)&255:02x}\n'for v in a.flat))
        weight.write_text(''.join(f'{int(v)&255:02x}\n'for v in w.flat))
        coeff.write_text(''.join(f'{int(x)&((1<<64)-1):016x} {int(y)&((1<<64)-1):016x}\n'for x,y in zip(m,b)))
        command=(['vvp',str(self.binary)]if self.simulator=='icarus'else[str(self.binary)])+[
            f'+WEIGHTS={weight}',f'+IMAGE={image}',f'+COEFFICIENTS={coeff}',f'+OUTPUT={output}',f'+SINK_PATTERN={int(sink_stalls)}']
        started=time.perf_counter_ns();result=subprocess.run(command,capture_output=True,text=True,timeout=1800)
        (root/'simulation.log').write_text(result.stdout+result.stderr);result.check_returncode()
        wall=time.perf_counter_ns()-started
        matches=re.findall(r'STEM_FRONTEND_DONE outputs=(\d+) latency=(\d+) slots=(\d+) stalls=(\d+)',result.stdout)
        if len(matches)!=1:raise ValueError('Missing actual frontend counters')
        count,elapsed,slots,stalls=map(int,matches[0])
        tokens=output.read_text().splitlines()
        if count!=expected.size or len(tokens)!=count or slots!=elapsed+1 or any(not re.fullmatch('[0-9a-fA-F]{2}',v)for v in tokens):
            raise ValueError('Malformed/incomplete frontend evidence')
        actual=np.asarray([int(v,16)for v in tokens],dtype=np.int64).reshape(expected.shape)
        if not np.array_equal(actual,expected):raise AssertionError('Actual connected stem RTL differs from independent reference')
        record=dict(index=index,site=site,result='PASS',protocol=self.identity['protocol'],
            geometry=geometry,input_sha256=tensor_sha(a),weights_sha256=tensor_sha(w),
            private_input_sha256=sha(image),private_weight_sha256=sha(weight),private_coefficients_sha256=sha(coeff),
            output_sha256=tensor_sha(actual),reference_sha256=tensor_sha(expected),checked_values=count,
            frontend_latency_cycles=elapsed,observation_slots=slots,output_stall_cycles=stalls,
            clock_period_ns=10,clock_scope='simulation only; not validated Fmax',
            interval='first image byte accepted to last pooled output accepted; model initialization excluded',
            host_simulator_wall_ns=wall,whole_network_cycles=None,full_network_rtl_pass=False,accuracy=False,board=False)
        (self.evidence/f'stem-{index:04d}.json').write_text(json.dumps(record,indent=2)+'\n')
        suite=ET.Element('testsuite',name='Connected INT8 stem RTL',tests='1',failures='0',errors='0',skipped='0')
        ET.SubElement(suite,'testcase',name=site,classname=self.identity['protocol'])
        ET.ElementTree(suite).write(self.evidence/f'junit-{index:04d}.xml',encoding='utf-8',xml_declaration=True)
        self.calls.append(record);return actual
