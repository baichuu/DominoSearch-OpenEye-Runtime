"""Actual Q40 output affine RTL; no inherited hybrid FP64 logit/accuracy claim."""
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
from fpga.openeye.rtl_activation import sha,tensor_sha
from fpga.openeye.runtime_activations import integer_array

FRACTION_BITS=40


def coefficients(logit_scale,bias):
    b=np.asarray(bias,dtype=np.float64)
    if b.ndim!=1 or not b.size or not np.all(np.isfinite(b)) or not np.isfinite(logit_scale) or logit_scale<=0:
        raise ValueError('Invalid fixed-point logit calibration')
    m=int(round(float(logit_scale)*2**FRACTION_BITS));wide=np.rint(b*2**FRACTION_BITS)
    if m<1 or m>=1<<63 or np.max(np.abs(wide))>=1<<63:raise ValueError('Q40 scale/bias outside INT64')
    return m,wide.astype(np.int64)


class LogitBackend:
    def __init__(self,private,evidence):
        self.private=Path(private);self.evidence=Path(evidence);self.calls=[]
        if self.private.exists() or self.evidence.exists():raise ValueError('New logit paths required')
        self.private.mkdir(parents=True);self.evidence.mkdir(parents=True)
        self.binary=self.private/'logits.vvp';rtl=Path(__file__).with_name('rtl')
        sources=[]
        for name in ('openeye_logit_affine_stream.sv','openeye_logit_affine_tb.sv','openeye_network_cycle_monitor.sv'):
            p=self.private/name;p.write_bytes((rtl/name).read_bytes());sources.append(p)
        result=subprocess.run(['iverilog','-g2012','-s','openeye_logit_affine_tb','-o',str(self.binary),*map(str,sources)],capture_output=True,text=True)
        (self.private/'compile.log').write_text(result.stdout+result.stderr);result.check_returncode()
        self.identity=dict(protocol='int64-q40-logit-affine-v1',fraction_bits=FRACTION_BITS,
            rtl_sha256={p.name:sha(p)for p in sources},binary_sha256=sha(self.binary),board=False)
        (self.evidence/'source-identity.json').write_text(json.dumps(self.identity,indent=2)+'\n')

    def run(self,accumulator,multiplier,bias,*,site='fc.bias',stalls=False):
        a=integer_array(accumulator,low=-(1<<31),high=(1<<31)-1)
        b=np.asarray(bias)
        if a.ndim!=1 or b.shape!=a.shape or b.dtype.kind!='i' or type(multiplier)is not int or not -(1<<63)<=multiplier<1<<63:
            raise ValueError('Invalid logit affine operands')
        values=[int(x)*multiplier+int(y)for x,y in zip(a,b)]
        if any(not -(1<<63)<=v<1<<63 for v in values):raise ValueError('INT64 logit overflow before DUT')
        expected=np.asarray(values,dtype=np.int64)
        if sha(self.binary)!=self.identity['binary_sha256']:raise ValueError('Changed logit binary')
        index=len(self.calls);root=self.private/f'command-{index:04d}';root.mkdir();incoming=root/'input.hex';output=root/'output.hex'
        incoming.write_text(''.join(f'{int(x)&((1<<32)-1):08x} {multiplier&((1<<64)-1):016x} {int(y)&((1<<64)-1):016x}\n'for x,y in zip(a,b)))
        result=subprocess.run(['vvp',str(self.binary),f'+INPUT={incoming}',f'+OUTPUT={output}',f'+COUNT={a.size}',f'+STALLS={int(stalls)}'],capture_output=True,text=True,timeout=30)
        (root/'simulation.log').write_text(result.stdout+result.stderr);result.check_returncode()
        matches=re.findall(r'LOGIT_DONE count=(\d+) latency=(\d+) slots=(\d+) stalls=(\d+)',result.stdout)
        if len(matches)!=1:raise ValueError('Missing logit counter evidence')
        count,elapsed,slots,blocked=map(int,matches[0]);tokens=output.read_text().splitlines()
        if count!=a.size or len(tokens)!=count or slots!=elapsed+1 or any(not re.fullmatch('[0-9a-fA-F]{16}',v)for v in tokens):raise ValueError('Malformed actual logits')
        words=[int(v,16)for v in tokens];actual=np.asarray([v-(1<<64)if v>>63 else v for v in words],dtype=np.int64)
        if not np.array_equal(actual,expected):raise AssertionError('RTL Q40 logit affine mismatch')
        record=dict(index=index,site=site,result='PASS',protocol=self.identity['protocol'],fraction_bits=FRACTION_BITS,
            checked_values=count,input_sha256=tensor_sha(a),output_sha256=tensor_sha(actual),reference_sha256=tensor_sha(expected),
            private_input_sha256=sha(incoming),private_output_sha256=sha(output),latency_cycles=elapsed,observation_slots=slots,
            output_stall_cycles=blocked,clock_period_ns=10,clock_scope='simulation only; no Fmax validation',
            whole_network_cycles=None,full_network_rtl_pass=False,accuracy=False,board=False)
        (self.evidence/f'logits-{index:04d}.json').write_text(json.dumps(record,indent=2)+'\n')
        suite=ET.Element('testsuite',name='Fixed-point logit affine RTL',tests='1',failures='0',errors='0',skipped='0')
        ET.SubElement(suite,'testcase',name=site,classname=self.identity['protocol'])
        ET.ElementTree(suite).write(self.evidence/f'junit-{index:04d}.xml',encoding='utf-8',xml_declaration=True)
        self.calls.append(record);return actual
