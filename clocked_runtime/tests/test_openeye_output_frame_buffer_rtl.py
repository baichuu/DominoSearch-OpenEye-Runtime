from pathlib import Path
import subprocess
import pytest


@pytest.mark.parametrize('length',[1,8,63])
def test_whole_frame_reserved_before_core_output_and_stalled_drain(tmp_path,length):
    rtl=Path(__file__).resolve().parents[1]/'fpga/openeye/rtl/openeye_dma_output_frame_buffer.sv'
    tb=tmp_path/'tb.sv';binary=tmp_path/'frame.vvp'
    tb.write_text('''module tb;
reg clk=0,rst=1,start=0,valid_in=0,ready_out=0;
always #5 clk=~clk;
reg[63:0]data_in=0;
wire command_ready,ready_in,valid_out,done,invalid;
wire[63:0]data_out;
integer written=0,read=0,cycle=0;reg blocked=0;reg[63:0]held;
openeye_dma_output_frame_buffer #(.CAPACITY(64)) dut(.clk_i(clk),.rst_ni(rst),.start_i(start),.command_ready_o(command_ready),
 .frame_words_i(32'd'''+str(length)+'''),.valid_i(valid_in),.ready_o(ready_in),.data_i(data_in),
 .valid_o(valid_out),.ready_i(ready_out),.data_o(data_out),.done_o(done),.invalid_o(invalid));
always @(negedge clk)if(rst)begin
 valid_in=written<'''+str(length)+''' && !command_ready;
 data_in=64'hfedc0123456789ab ^ written;
 ready_out=cycle%23>12 && cycle%3!=0;
end
always @(posedge clk)if(rst)begin
 cycle=cycle+1;
 if(invalid)$fatal(1,"Unexpected frame error");
 if(valid_in && written<'''+str(length)+''' && !ready_in)$fatal(1,"Core stalled inside reserved frame");
 if(valid_in && ready_in)written<=written+1;
 if(valid_out && written!='''+str(length)+''')$fatal(1,"Drain before entire frame captured");
 if(blocked && (!valid_out || data_out!==held))$fatal(1,"Frame data changed under stall");
 blocked<=valid_out && !ready_out;held<=data_out;
 if(valid_out && ready_out)begin
  if(data_out!==(64'hfedc0123456789ab ^ read))$fatal(1,"Frame output lost/duplicated");
  read<=read+1;
 end
 if(done)begin
  if(read!='''+str(length)+''' || !command_ready)$fatal(1,"Incomplete frame");
  $display("FRAME_PASS");$finish;
 end
 if(cycle>1000)$fatal(1,"Frame timeout");
end
initial begin #1;rst=0;#1;rst=1;@(negedge clk);start=1;@(negedge clk);start=0;end
endmodule
''')
    subprocess.run(['iverilog','-g2012','-s','tb','-o',str(binary),str(rtl),str(tb)],capture_output=True,check=True)
    result=subprocess.run(['vvp',str(binary)],capture_output=True,text=True,check=True)
    assert 'FRAME_PASS'in result.stdout
