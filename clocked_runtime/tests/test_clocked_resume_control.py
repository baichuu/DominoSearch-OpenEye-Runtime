"""Synthetic RAM/FSM save-restore controls; not native OpenEye evidence."""
from pathlib import Path
import json
import subprocess
import shutil
import pytest
from fpga.openeye.resumable_graph_top import PORTS


def test_three_processes_preserve_ram_clock_and_every_trace_byte(tmp_path):
    if not shutil.which('verilator'):pytest.skip('Verilator required')
    top=tmp_path/'mock.sv'
    top.write_text(PORTS+'''
reg[63:0]count=0;reg[63:0]memory[0:3];integer i;
always @(posedge clk or negedge rst)begin
 if(!rst)begin count<=0;for(i=0;i<4;i=i+1)memory[i]<=i+17;end
 else begin
  count<=count+1;memory[count%4]<=memory[(count+1)%4]+count+3;
  if(count==120)$finish;
 end
end
assign input_fire_o=rst && count%3==0;assign input_word_o=memory[count%4];
assign raw_fire_o=rst && count%5==0;assign raw_word_o=memory[(count+2)%4];
assign forward_fire_o=rst && count%4==0;assign forward_word_o=memory[(count+3)%4];
assign forward_address_o=count;assign instruction_o=count/10;assign completed_o=count/10;
assign last_logged_o=count/10;assign stem_count_o=0;assign stem_boundary_o=0;assign stem_fire_o=0;assign stem_word_o=0;
assign final_fire_o=rst && count%7==0;assign cycles_o=count;assign engine_cycles_o=count;
assign elapsed_o=count;assign setup_done_o=1;
endmodule
''')
    cpp=Path(__file__).resolve().parents[1]/'fpga/openeye/rtl/clocked_resume_main.cpp'
    subprocess.run(['verilator','--cc','--exe','--build','--savable','--top-module','openeye_graph_resumable','--Mdir',str(tmp_path/'obj'),
        '-o','resume-test','-j','2','-Wno-fatal','-CFLAGS','-std=c++17',str(top),str(cpp)],capture_output=True,text=True,check=True)
    binary=tmp_path/'obj/resume-test';whole=tmp_path/'whole';split=tmp_path/'split';whole.mkdir();split.mkdir()
    def invoke(root,*args):subprocess.run([str(binary),*args],cwd=root,capture_output=True,text=True,check=True)
    invoke(whole)
    invoke(split,'--stop-after-edges','23','--save',str(split/'first.model'))
    invoke(split,'--resume',str(split/'first.model'),'--stop-after-edges','41','--save',str(split/'second.model'))
    invoke(split,'--resume',str(split/'second.model'))
    for name in('actual-accepted.hex','actual-raw.hex','actual-forwarded.hex','actual-output.hex','actual-stem.hex'):
        assert (split/name).read_bytes()==(whole/name).read_bytes(),name
    assert json.loads((split/'resume-status.json').read_text())==json.loads((whole/'resume-status.json').read_text())
