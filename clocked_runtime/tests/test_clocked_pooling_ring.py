"""Inactive-pooling isolation controls on actual upstream ring-buffer RTL."""
import os
from pathlib import Path
import subprocess
import pytest
from fpga.openeye.clocked_core_compat import gate_unused_pooling_ring


@pytest.mark.parametrize('isolate',[False,True])
def test_geometry_change_with_inactive_pooling_cannot_write_ring(tmp_path,isolate):
    checkout=Path(os.environ.get('OPENEYE_NEW_PIN_ROOT',Path.home()/'Projects/OpenEye-9a742fb'))
    tb=tmp_path/'tb.sv';tb.write_text('''module tb;
reg clk=0,rst=1;always #5 clk=~clk;
reg[3:0]limit=7;reg ready=0,pooling=0;wire[7:0]data;wire valid;
ring_buffer_pipelined #(.ADDR_WIDTH(4),.DATA_WIDTH(8)) dut(.clk_i(clk),.rst_n(rst),
 .limit_i(limit),.set_pointer_start_i(1'b0),.ready_i('''+('ready && pooling'if isolate else'ready')+'''),
 .data_i(8'd0),.data_o(data),.valid_o(valid));
initial begin #1;rst=0;repeat(2)@(negedge clk);rst=1;ready=1;
 repeat(6)@(negedge clk);limit=1;repeat(40)@(negedge clk);
 $display("INACTIVE_POOLING_ISOLATED");$finish;end
endmodule
''')
    paths=[checkout/'hdl'/name for name in('ring_buffer_pipelined.v','RAM_DP.v','RAM_DP_generic.v')]
    binary=tmp_path/'ring.vvp';subprocess.run(['iverilog','-g2012','-s','tb','-o',str(binary),str(tb),*map(str,paths)],check=True,capture_output=True)
    result=subprocess.run(['vvp',str(binary)],capture_output=True,text=True)
    collision='R/W collision'in result.stdout+result.stderr
    assert collision is (not isolate)
    if isolate:assert result.returncode==0 and 'INACTIVE_POOLING_ISOLATED'in result.stdout


def test_generated_binding_patch_is_exact_and_idempotent(tmp_path):
    hdl=tmp_path/'hdl';hdl.mkdir();p=hdl/'OpenEye_FPGA.v';p.write_text(' .ready_i(pooling_buffer_enable),\n')
    gate_unused_pooling_ring(tmp_path);gate_unused_pooling_ring(tmp_path)
    assert p.read_text()==' .ready_i(pooling_buffer_enable && pooling),\n'
    p.write_text('unknown source')
    with pytest.raises(ValueError):gate_unused_pooling_ring(tmp_path)
