"""Actual clock-edge controls for the observer, not network DUT evidence."""
from pathlib import Path
import shutil
import subprocess

import pytest

RTL = Path(__file__).resolve().parents[1] / 'fpga/openeye/rtl/openeye_network_cycle_monitor.sv'


def simulate(tmp_path, rows, width=64):
    if not shutil.which('iverilog') or not shutil.which('vvp'):
        pytest.skip('Icarus required for actual cycle monitor test')
    # bits: start valid/ready, finish valid/ready, compute/postops/transfer.
    stimulus = '\n'.join('reset_observer;' if row=='RESET' else f"drive(7'b{row});" for row in rows)
    tb = tmp_path / 'tb.sv'
    tb.write_text('''module tb;
reg clk=0, rst=1;
reg [6:0] pins=0;
wire running, done, invalid;
parameter W=''' + str(width) + ''';
wire [W-1:0] total, compute, postops, transfer, idle, overlap, elapsed;
always #5 clk=~clk;
openeye_network_cycle_monitor #(.COUNTER_WIDTH(W)) dut (
 .clk_i(clk),.rst_ni(rst),
 .start_valid_i(pins[6]),.start_ready_i(pins[5]),
 .finish_valid_i(pins[4]),.finish_ready_i(pins[3]),
 .compute_active_i(pins[2]),.postops_active_i(pins[1]),.transfer_active_i(pins[0]),
 .running_o(running),.done_o(done),.invalid_o(invalid),
 .elapsed_cycles_o(elapsed),
 .total_cycles_o(total),.compute_cycles_o(compute),.postops_cycles_o(postops),
 .transfer_cycles_o(transfer),.idle_cycles_o(idle),.overlap_cycles_o(overlap));
task drive(input [6:0] row);
begin
 @(negedge clk); pins=row;
 @(posedge clk); #1;
 $display("OBS %d %d %d %d %d %d %d %d %d %d",running,done,invalid,total,compute,postops,transfer,idle,overlap,elapsed);
end
endtask
task reset_observer;
begin
 @(negedge clk); rst=0; pins=0;
 @(posedge clk); #1;
 $display("OBS %d %d %d %d %d %d %d %d %d %d",running,done,invalid,total,compute,postops,transfer,idle,overlap,elapsed);
 @(negedge clk); rst=1;
end
endtask
initial begin
 #1; rst=0; #1; rst=1;
''' + stimulus + '''
 $finish;
end
endmodule
''')
    binary=tmp_path/'monitor.vvp'
    subprocess.run(['iverilog','-g2012','-s','tb','-o',str(binary),str(RTL),str(tb)],check=True,capture_output=True,text=True)
    result=subprocess.run(['vvp',str(binary)],check=True,capture_output=True,text=True)
    return [tuple(map(int,line.split()[1:])) for line in result.stdout.splitlines() if line.startswith('OBS ')]


def test_continuous_edges_include_waits_and_overlap_once(tmp_path):
    trace=simulate(tmp_path,[
        '1000111',  # start blocked: no transaction despite activity
        '1100001',  # first input accepted, transfer
        '0000111',  # three units overlap, ONE elapsed cycle
        '0000000',  # waiting with no activity still counts
        '0010110',  # output valid but sink blocked
        '0011001',  # last output accepted
        '0000111',  # outside interval, frozen
    ])
    assert trace[0]==(0,0,0,0,0,0,0,0,0,0)
    assert trace[-2]==(0,1,0,5,2,2,3,1,2,4)
    assert trace[-1]==trace[-2]


def test_same_edge_and_early_finish_are_distinguished(tmp_path):
    assert simulate(tmp_path,['1111000'])[-1]==(0,1,0,1,0,0,0,1,0,0)
    sub=tmp_path/'early';sub.mkdir()
    assert simulate(sub,['0011000'])[-1]==(0,0,1,0,0,0,0,0,0,0)


@pytest.mark.parametrize('rows',[
    ['1100000','1100000','0011000'],  # second start during execution
    ['1111000','1100000'],           # second run without reset
    ['1111000','0011000'],           # repeated final output boundary
])
def test_repeated_boundaries_invalidate_measurement(tmp_path,rows):
    assert simulate(tmp_path,rows)[-1][2]==1


def test_counter_overflow_saturates_and_invalidates(tmp_path):
    rows=['1100100']+['0000100']*7+['0011100']
    trace=simulate(tmp_path,rows,width=3)
    assert trace[-1][0:4]==(0,1,1,7)
    assert trace[-1][4]==7


def test_reset_aborts_partial_measurement_and_clears_prior_error(tmp_path):
    trace=simulate(tmp_path,['1100100','1100100','RESET','0000000','1111010'])
    assert trace[1][2]==1  # invalid repeated start before reset
    assert trace[2]==trace[3]==(0,0,0,0,0,0,0,0,0,0)
    assert trace[-1]==(0,1,0,1,0,1,0,0,0,0)
