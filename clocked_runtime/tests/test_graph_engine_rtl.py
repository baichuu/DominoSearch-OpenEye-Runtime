"""Clocked controller integration controls with a mock OpenEye core."""
from pathlib import Path
import subprocess
import pytest

from test_openeye_relu_rtl import openeye
from fpga.openeye.rtl_activation import ReLUBackend


def instruction(op, *args):
    fields = [op, *args]+[0]*(31-len(args))
    return ''.join(f'{v & 0xffffffff:08x}' for v in reversed(fields))+'\n'


@pytest.mark.parametrize('stall', [0, 1])
@pytest.mark.parametrize('pipelined', [0, 1])
def test_scatter_channel_sum_affine_residual_gap_logit(openeye, tmp_path, stall,pipelined):
    rtl = Path(__file__).resolve().parents[1]/'fpga/openeye/rtl'
    relu = ReLUBackend(openeye, tmp_path/'relu', tmp_path/'relu-evidence')
    program = tmp_path/'program.hex'
    program.write_text(
        instruction(1, 0, 2, 0, 0, 0, 2, 0, 0)+
        instruction(1, 0, 2, 0, 0, 0, 2, 0, 1)+
        instruction(2, 0, 16, 4, 0, 2, 1, 0)+
        instruction(3, 16, 24, 4, 2, 0, 16, 1)+
        instruction(4, 24, 32, 4, 2)+
        instruction(5, 32, 40, 2, 3)+instruction(0))
    tb = tmp_path/'tb.sv'
    tb.write_text('''module tb;
reg clk=0,rst=1,start=0;always #5 clk=~clk;
reg[1023:0]program_rom[0:6];reg signed[63:0]memory[0:63];reg signed[63:0]read_data;
reg[31:0]maps[0:3];reg signed[63:0]m[0:4],b[0:4];
wire[31:0]ip,packet,map,coefficient,address,completed,accepted,raws;
wire rd,wr,en,ready,core_valid,core_sink,done,invalid,ins_done;
wire[63:0]payload,raw;wire signed[63:0]write_data;wire[63:0]cycles;
reg enable_q=0;integer cycle=0,incoming=0,outgoing=0,command=0,completed_count=0;
reg sending=0,ready_core=1;
assign ready=ready_core && !sending;
assign core_valid=sending;
assign raw=outgoing==0 ? 64'hfffffffdfffffffe : 64'h0000000500000004;
openeye_graph_engine #(.FRAME_CAPACITY(8),.POSTOPS_PIPELINED('''+str(pipelined)+''')) dut(
 .clk_i(clk),.rst_ni(rst),.start_i(start),.command_ready_o(),.instruction_address_o(ip),.instruction_i(program_rom[ip]),
 .packet_address_o(packet),.packet_data_i(64'd0),.map_address_o(map),.map_data_i(maps[map]),
 .coefficient_address_o(coefficient),.coefficient_multiplier_i(m[coefficient]),.coefficient_bias_i(b[coefficient]),
 .memory_read_o(rd),.memory_write_o(wr),.memory_address_o(address),.memory_data_i(read_data),.memory_data_o(write_data),
 .enable_dma_o(en),.data_dma_o(payload),.ready_dma_i(ready),.core_valid_i(core_valid),.core_data_i(raw),.core_ready_o(core_sink),
 .done_o(done),.invalid_o(invalid),.instruction_done_o(ins_done),.completed_instruction_o(completed),
 .accepted_words_o(accepted),.raw_words_o(raws),.elapsed_cycles_o(cycles));
always @(negedge clk)ready_core='''+('1' if not stall else 'cycle%11>4')+''';
always @(posedge clk)begin
 if(rd)read_data<=memory[address];if(wr)memory[address]<=write_data;
 if(!rst)begin enable_q<=0;incoming<=0;sending<=0;outgoing<=0;command<=0;end
 else begin
  enable_q<=en;cycle=cycle+1;
  if(cycle>10000 || invalid)$fatal(1,"Graph controller failure");
  if(enable_q && ready)begin
   if(incoming==1)begin incoming<=0;sending<=1;end else incoming<=incoming+1;
  end
  if(core_valid && core_sink)begin
   if(outgoing==1)begin outgoing<=0;sending<=0;command<=command+1;end else outgoing<=outgoing+1;
  end
  if(ins_done)begin
   if(completed!=completed_count)$fatal(1,"Lost instruction");completed_count<=completed_count+1;
  end
  if(done)begin
   if(memory[0]!=-6 || memory[1]!=10 || memory[2]!=8 || memory[3]!=-4)$fatal(1,"Wrong raw scatter sum");
   if(memory[16]!=0 || memory[17]!=21 || memory[18]!=0 || memory[19]!=13)$fatal(1,"Wrong affine/ReLU");
   if(memory[24]!=0 || memory[25]!=42 || memory[26]!=0 || memory[27]!=26)$fatal(1,"Wrong residual alignment");
   if(memory[32]!=21 || memory[33]!=13)$fatal(1,"Wrong GAP");
   if(memory[40]!=61 || memory[41]!=47)$fatal(1,"Wrong logit bias");
   if(accepted!=4 || raws!=4 || completed_count!=6)$fatal(1,"Wrong observed counters");
   $display("GRAPH_MOCK_PASS cycles=%0d",cycles);$finish;
  end
 end
end
initial begin
 $readmemh("'''+str(program)+'''",program_rom);
 maps[0]=3;maps[1]=0;maps[2]=2;maps[3]=1;
 m[0]=2;b[0]=1;m[1]=-2;b[1]=5;m[2]=1;b[2]=3;
 m[3]=3;b[3]=-2;m[4]=4;b[4]=-5;
 #1;rst=0;#1;rst=1;repeat(4)@(negedge clk);start=1;@(negedge clk);start=0;
end
endmodule
''')
    names = ['openeye_graph_engine.sv', 'openeye_registered_dma_sender.sv', 'openeye_dma_output_frame_buffer.sv',
             'openeye_activation_dma_stream.sv', 'openeye_postops_elastic.sv', 'openeye_postops_stream.sv',
             'openeye_gap_elastic.sv', 'openeye_logit_affine_stream.sv']
    binary = tmp_path/'graph.vvp'
    subprocess.run(['iverilog', '-g2012', '-s', 'tb', '-o', str(binary), str(tb), str(relu.private/'af_cluster.v'),
                    *[str(rtl/n) for n in names]], check=True, capture_output=True)
    result = subprocess.run(['vvp', str(binary)], check=True, capture_output=True, text=True, timeout=20)
    assert 'GRAPH_MOCK_PASS' in result.stdout
