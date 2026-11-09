"""Actual postops/controller/SRAM controls; native convolution is not exercised."""
import json
from pathlib import Path
import re
import subprocess
import random
import pytest
from test_openeye_relu_rtl import openeye
from test_graph_engine_rtl import instruction
from fpga.openeye.rtl_activation import ReLUBackend

RTL=Path(__file__).resolve().parents[1]/'fpga/openeye/rtl'


def rounded(z,shift):
    q,r=divmod(z,1<<shift)
    return q+int(2*r>1<<shift or (2*r==1<<shift and q%2))


def run_postop(root,af,*,values,skip,multipliers,biases,op,shift,relu,pixels,pipelined,depth,alias='none',offset_overlap=False,reset_midstream=False,simulator='iverilog'):
    root.mkdir();count=len(values);src=8;skipbase=src if alias=='both'else src+count+16
    dst={'main':src,'skip':skipbase,'both':src}.get(alias,skipbase+count+16)
    if offset_overlap:dst=(skipbase if offset_overlap=='skip'else src)+1
    memory=[0]*(dst+count+16 if dst>=skipbase else skipbase+count+16)
    memory[src:src+count]=values;memory[skipbase:skipbase+count]=skip
    actual_values=memory[src:src+count];actual_skip=memory[skipbase:skipbase+count]
    expected=[]
    for i,(a,b)in enumerate(zip(actual_values,actual_skip)):
        c=i//pixels if op==2 else 0
        z=a*multipliers[c]+(b*biases[c]if op==3 else biases[c])
        assert -(1<<63)<=z<(1<<63)
        q=rounded(z,shift)
        if relu or op==3:q=max(0,q)
        expected.append(min(127,max(-127,q)))
    def words(path,seq):path.write_text(''.join(f'{v&((1<<64)-1):016x}\n'for v in seq))
    words(root/'memory.hex',memory);words(root/'m.hex',multipliers);words(root/'b.hex',biases)
    args=(src,dst,count,0,pixels,int(relu),shift)if op==2 else(src,dst,count,0,0,skipbase,shift)
    (root/'program.hex').write_text(instruction(op,*args)+instruction(0))
    tb='''module tb;
reg clk=0,rst=1,start=0;always #5 clk=~clk;
reg[1023:0]program_rom[0:1];reg signed[63:0]memory[0:MEMORY-1],m[0:COEFF-1],b[0:COEFF-1],read_data;
wire[31:0]ip,coefficient,address;wire rd,wr,done,invalid;wire signed[63:0]data;wire[63:0]cycles;
integer ticks=0,writes=0,file_out;
openeye_graph_engine #(.FRAME_CAPACITY(8),.POSTOPS_PIPELINED(PIPE),.POSTOPS_FIFO_DEPTH(DEPTH)) dut(
 .clk_i(clk),.rst_ni(rst),.start_i(start),.command_ready_o(),.instruction_address_o(ip),.instruction_i(program_rom[ip]),
 .packet_address_o(),.packet_data_i(64'd0),.map_address_o(),.map_data_i(32'd0),
 .coefficient_address_o(coefficient),.coefficient_multiplier_i(m[coefficient]),.coefficient_bias_i(b[coefficient]),
 .memory_read_o(rd),.memory_write_o(wr),.memory_address_o(address),.memory_data_i(read_data),.memory_data_o(data),
 .enable_dma_o(),.data_dma_o(),.ready_dma_i(1'b1),.core_valid_i(1'b0),.core_data_i(64'd0),.core_ready_o(),
 .done_o(done),.invalid_o(invalid),.instruction_done_o(),.completed_instruction_o(),
 .accepted_words_o(),.raw_words_o(),.elapsed_cycles_o(cycles));
always @(posedge clk)if(!rst)begin ticks=0;writes=0;end else begin
 ticks=ticks+1;
 if(ticks>LIMIT)$fatal(1,"Postops timeout");
 if(invalid)begin
   if(REJECT && writes==0)begin $display("OVERLAP_REJECTED");$finish;end
   $fatal(1,"Unexpected controller invalid");
 end
 if(rd && wr)$fatal(1,"Candidate assumed two scratch addresses");
 if((rd || wr) && address>=MEMORY)$fatal(1,"Memory boundary");
 if(rd)read_data<=memory[address];
 if(wr)begin
  if(address!=DEST+writes || writes>=COUNT)$fatal(1,"Lost/reordered/duplicate output");
  memory[address]<=data;$fwrite(file_out,"%016h\\n",data);writes=writes+1;
 end
 if(done)begin
  if(REJECT || writes!=COUNT)$fatal(1,"Incomplete instruction");
  $fclose(file_out);$display("POSTOPS_CONTROLLER_PASS cycles=%0d writes=%0d",cycles,writes);$finish;
 end
end
initial begin
 $readmemh("MEMFILE",memory);$readmemh("MFILE",m);$readmemh("BFILE",b);$readmemh("PROGFILE",program_rom);
 file_out=$fopen("OUTFILE","w");#1;rst=0;#1;rst=1;repeat(4)@(negedge clk);start=1;@(negedge clk);start=0;
 RESET_CONTROL
end
endmodule
'''
    replacements={'MEMORY':len(memory),'COEFF':len(multipliers),'PIPE':pipelined,'DEPTH':depth,'LIMIT':count*20+200,
        'DEST':dst,'COUNT':count,'REJECT':int(bool(offset_overlap)),'MEMFILE':root/'memory.hex','MFILE':root/'m.hex',
        'BFILE':root/'b.hex','PROGFILE':root/'program.hex','OUTFILE':root/'output.hex'}
    replacements['RESET_CONTROL']=('wait(dut.post.core.valid_s2);@(negedge clk);rst=0;repeat(2)@(negedge clk);rst=1;'
        'repeat(2)@(negedge clk);start=1;@(negedge clk);start=0;'if reset_midstream else'')
    for key,value in replacements.items():tb=re.sub(r'\b'+key+r'\b',lambda _:str(value),tb)
    (root/'tb.sv').write_text(tb)
    modules=['openeye_graph_engine.sv','openeye_registered_dma_sender.sv','openeye_dma_output_frame_buffer.sv',
        'openeye_activation_dma_stream.sv','openeye_postops_elastic.sv','openeye_postops_stream.sv',
        'openeye_gap_elastic.sv','openeye_logit_affine_stream.sv']
    sources=[str(root/'tb.sv'),str(af),*[str(RTL/name)for name in modules]]
    if simulator=='iverilog':command=['iverilog','-g2012','-s','tb','-o',str(root/'test.vvp'),*sources]
    else:
        assert simulator=='verilator'
        command=['verilator','--binary','--timing','--top-module','tb','--Mdir',str(root/'obj'),'-j','2','-Wno-fatal',*sources]
    compiled=subprocess.run(command,capture_output=True,text=True)
    assert compiled.returncode==0,compiled.stderr
    run=subprocess.run(['vvp',str(root/'test.vvp')]if simulator=='iverilog'else[str(root/'obj/Vtb')],
        capture_output=True,text=True,timeout=max(60,count//512))
    assert run.returncode==0,run.stdout+run.stderr
    if offset_overlap:
        assert 'OVERLAP_REJECTED'in run.stdout;return None
    actual=[int(v,16)for v in(root/'output.hex').read_text().splitlines()]
    actual=[v-(1<<64)if v>>63 else v for v in actual];assert actual==expected
    found=re.search(r'POSTOPS_CONTROLLER_PASS cycles=(\d+) writes=(\d+)',run.stdout);assert found
    assert int(found[2])==count
    return int(found[1])


@pytest.mark.parametrize('op,count,depth,shift,relu,pixels,alias',[
    (2,1,1,0,False,1,'none'),(2,2,1,1,True,1,'main'),(2,9,3,2,False,3,'none'),
    (2,64,8,24,True,8,'main'),(2,257,8,40,False,1,'none'),(2,4096,8,24,True,64,'none'),
    (3,1,1,0,True,1,'both'),(3,3,1,1,True,1,'main'),(3,31,3,2,True,1,'skip'),
    (3,64,8,24,True,1,'none'),(3,257,8,40,True,1,'both'),(3,4096,8,24,True,1,'skip')])
def test_pipeline_matches_independent_arithmetic_and_legacy_controller(openeye,tmp_path,record_property,op,count,depth,shift,relu,pixels,alias):
    af=ReLUBackend(openeye,tmp_path/'relu',tmp_path/'evidence').private/'af_cluster.v'
    rng=random.Random(700+count+op)
    values=[rng.randrange(-(1<<30),1<<30)for _ in range(count)]
    skip=[rng.randrange(-127,128)for _ in range(count)]
    coefficients=(count+pixels-1)//pixels if op==2 else 1
    m=[rng.randrange(-(1<<20),1<<20)for _ in range(coefficients)]
    b=[rng.randrange(-(1<<20),1<<20)for _ in range(coefficients)]
    if count>=9:
        values[:9]=[0,1,-1,2,-2,127,-127,128,-128]
        # Deliberately exercise half-way ties with positive/negative products.
        m[0]=1;b[0]=0 if op==2 else 1
    cycles=[]
    for pipe in (0,1):
        cycles.append(run_postop(tmp_path/str(pipe),af,values=values,skip=skip,multipliers=m,biases=b,op=op,
            shift=shift,relu=relu,pixels=pixels,pipelined=pipe,depth=depth,alias=alias))
    assert cycles[1]<=cycles[0]
    if count>=64 and depth==8:assert cycles[1]<0.6*cycles[0]
    record_property('baseline_cycles',cycles[0]);record_property('candidate_cycles',cycles[1]);record_property('checked_values',count)
    record_property('op',op);record_property('fifo_depth',depth)


@pytest.mark.parametrize('op,overlap',[(2,'main'),(3,'main'),(3,'skip')])
def test_shifted_overlap_rejected_before_any_write(openeye,tmp_path,op,overlap):
    af=ReLUBackend(openeye,tmp_path/'relu',tmp_path/'evidence').private/'af_cluster.v'
    run_postop(tmp_path/'reject',af,values=[1,2,3],skip=[4,5,6],multipliers=[1],biases=[0],op=op,shift=0,
        relu=True,pixels=3,pipelined=1,depth=8,offset_overlap=overlap)


@pytest.mark.parametrize('op,depth',[(2,1),(2,8),(3,1),(3,8)])
def test_reset_with_values_in_flight_has_no_stale_or_duplicate_output(openeye,tmp_path,op,depth):
    af=ReLUBackend(openeye,tmp_path/'relu',tmp_path/'evidence').private/'af_cluster.v'
    args=dict(values=[-7,-5,-3,-1,0,1,3,5,7,300,-300],skip=[1]*11,multipliers=[1],biases=[0 if op==2 else 1],
        op=op,shift=1,relu=False,pixels=11,pipelined=1,depth=depth)
    continuous=run_postop(tmp_path/'continuous',af,**args)
    reset=run_postop(tmp_path/'reset',af,reset_midstream=True,**args)
    assert reset==continuous
