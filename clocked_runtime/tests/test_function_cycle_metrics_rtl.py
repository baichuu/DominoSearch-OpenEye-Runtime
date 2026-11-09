"""Passive timing on actual fused RTL; operands and independent goldens stay local."""
from pathlib import Path
import re
import subprocess
import numpy as np
import pytest
from test_openeye_relu_rtl import openeye
from fpga.openeye.rtl_activation import ReLUBackend

RTL=Path(__file__).resolve().parents[1]/'fpga/openeye/rtl'


def records(log):
    return {int(d['function']):d for d in(
        {k:int(v)for k,v in re.findall(r'(\w+)=(\d+)',line)}
        for line in log.splitlines()if line.startswith('GRAPH_FUNCTION_METRIC '))}


@pytest.mark.parametrize('depth,stalls,pixels',[(1,False,0),(1,True,0),(3,True,0),(8,True,0),(8,True,49)])
def test_observers_preserve_every_output_and_cycles_with_bubbles_and_backpressure(openeye,tmp_path,depth,stalls,pixels):
    relu=ReLUBackend(openeye,tmp_path/'relu',tmp_path/'relu-evidence')
    rng=np.random.default_rng(500+depth);count=196 if pixels else 200;expected=[];operands=[];residuals=[];relus=[]
    for i in range(count):
        a,b,m,bias,n=map(int,rng.integers(-1000,1001,5));shift=(0,1,2,24,40)[i%5]
        residual=i%2;enabled=bool(pixels or i%3!=0);residuals.append(residual);relus.append(enabled)
        z=a*m+(b*n if residual else bias);q,r=divmod(z,1<<shift)
        q+=int(2*r>1<<shift or(2*r==1<<shift and q%2));q=max(q,0)if enabled else q
        expected.append(min(127,max(-127,q)))
        operands.append(' '.join(f'{v&((1<<64)-1):016x}'for v in(a,b,m,bias,n))+f' {shift} {residual} {int(enabled)}\n')
    incoming=tmp_path/'operands.hex';incoming.write_text(''.join(operands));results=[]
    if pixels:
        expected=[int(round(sum(expected[i:i+pixels])/pixels))for i in range(0,count,pixels)]
    for observed in(False,True):
        source=(RTL/'openeye_postops_elastic_tb.sv').read_text()
        if observed:
            assert source.count('$finish;')==1
            source=source.replace('  reg clk=0, rst=1;','  reg clk=0, rst=1;reg metric_dump=0;')
            source=source.replace('$finish;','@(negedge clk);metric_dump=1;@(negedge clk);$finish;')
            extra='''
openeye_postops_cycle_metrics #(.SCOPE(1)) metrics(
 .clk_i(clk),.rst_ni(rst),.clear_i(1'b0),.dump_i(metric_dump),.instruction_i(32'd7),
 .valid_i(post.valid_i),.ready_i(post.ready_o),.residual_i(post.residual_i),
 .valid_s1_i(post.core.valid_s1),.valid_s2_i(post.core.valid_s2),.valid_s3_i(post.core.valid_s3),
 .relu_s2_i(post.core.relu_s2),.core_valid_i(post.core.valid_o),
 .output_valid_i(post.valid_o),.output_ready_i(post.ready_i));
'''
            if pixels:
                extra+='''
openeye_function_cycle_observer #(.SCOPE(1),.FUNCTION_ID(8)) gap_metrics(
 .clk_i(clk),.rst_ni(rst),.clear_i(1'b0),.dump_i(metric_dump),.instruction_i(32'd7),
 .input_valid_i(WITH_GAP.gap.valid_i && WITH_GAP.gap.ready_o && WITH_GAP.gap.index==0),.input_ready_i(1'b1),
 .output_valid_i(output_valid),.output_ready_i(sink_ready));
'''
            source=source.rsplit('endmodule',1)[0]+extra+'endmodule\n'
        tb=tmp_path/f'tb-{observed}.sv';tb.write_text(source);binary=tmp_path/f'test-{observed}.vvp'
        modules=['openeye_postops_stream.sv','openeye_postops_elastic.sv','openeye_gap_elastic.sv',
                 'openeye_psum_accumulator_stream.sv','openeye_network_cycle_monitor.sv']
        if observed:modules+=['openeye_function_cycle_observer.sv','openeye_postops_cycle_metrics.sv']
        subprocess.run(['iverilog','-g2012','-s','openeye_postops_elastic_tb',
            f'-Popeneye_postops_elastic_tb.FIFO_DEPTH={depth}',f'-Popeneye_postops_elastic_tb.GAP_PIXELS={pixels}',
            '-o',str(binary),str(relu.private/'af_cluster.v'),
            *[str(RTL/n)for n in modules],str(tb)],check=True,capture_output=True,text=True)
        out=tmp_path/f'output-{observed}.hex'
        run=subprocess.run(['vvp',str(binary),f'+INPUT={incoming}',f'+OUTPUT={out}',f'+COUNT={count}',
            f'+SINK_PATTERN={int(stalls)}','+SOURCE_PATTERN=1'],check=True,capture_output=True,text=True)
        actual=[int(v,16)for v in out.read_text().splitlines()];actual=[v-(1<<64)if v>>63 else v for v in actual]
        assert actual==expected
        match=re.search(r'ELASTIC_DONE inputs=.*',run.stdout);assert match
        results.append((out.read_bytes(),match[0]))
        if observed:
            rows=records(run.stdout);assert set(rows)==set(range(1,9 if pixels else 8))
            sizes=[count-sum(residuals),sum(residuals),count,sum(relus),count-sum(relus),count,count]
            for function,size in enumerate(sizes,1):
                row=rows[function];assert row['inputs']==row['outputs']==size
                assert row['instruction']==7 and row['span']==row['last']-row['first']
                if function<7:
                    assert row['latency_min']==row['latency_max']==(1 if size else 0)and row['latency_sum']==size
                else:
                    assert row['latency_min']>=1 and row['latency_sum']>=size
                    if stalls:assert row['output_stalls']>0 and row['latency_max']>1
            if pixels:
                assert rows[8]['inputs']==rows[8]['outputs']==count//pixels
                assert rows[8]['latency_min']>=pixels
    assert results[0]==results[1]
