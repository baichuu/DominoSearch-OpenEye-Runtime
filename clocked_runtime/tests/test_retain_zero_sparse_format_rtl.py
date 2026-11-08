"""Actual activation decoder regression for the pin9a retention fallback."""
from pathlib import Path
import os
import subprocess
import pytest

from fpga.openeye.experiments.retain_zero_sparse_format import apply


@pytest.mark.parametrize('values', [[0]*12, [1,0,3,0,0,6,0,8,0,0,11,0], list(range(12))])
def test_retained_values_keep_sparse_position_and_address_metadata(tmp_path,values):
    checkout=Path(os.environ.get('OPENEYE_NEW_PIN_ROOT',Path(__file__).resolve().parents[2]/'OpenEye-9a742fb'))
    hdl=tmp_path/'hdl';hdl.mkdir()
    for name in ('PE.v','data_pipeline_iact.v'):
        source=subprocess.check_output(['git','-C',str(checkout),'show','9a742fbd42a55ebfa8fcd39aa41dfa36a367572f:hdl/'+name])
        (hdl/name).write_bytes(source)
    apply(tmp_path)
    drive=[]
    for i in range(0,len(values),2):
        packed=(values[i]&255)|((values[i+1]&255)<<12)
        drive.append(f'data=24\'d{packed};repeat(2)@(negedge clk);')
    tb=tmp_path/'tb.sv';tb.write_text('''module tb;
reg clk=0,rst=0,enable=0;always #5 clk=~clk;reg[23:0]data=0;
wire value_en,address_en;wire[11:0]payload;wire[3:0]address_data;wire[4:0]count;
data_pipeline_iact #(.SPARSITY_EN(1),.RETAIN_ZERO_VALUES(1)) dut(
 .clk_i(clk),.rst_ni(rst),.compute_i(1'b0),.data_i(data),.enable_i(enable),
 .iact_x_line_repetitions_i(4'd1),.first_spad_max_i(4'd3),
 .first_spad_en_o(address_en),.first_spad_data_o(address_data),
 .second_spad_en_o(value_en),.second_spad_data_o(payload),.second_spad_words_o(count));
integer addresses=0;
always @(posedge clk)begin
 if(value_en)$display("VALUE %0d",payload);
 if(address_en)begin addresses=addresses+1;$display("ADDRESS %0d",address_data);end
end
initial begin repeat(2)@(negedge clk);rst=1;@(negedge clk);enable=1;
'''+ '\n'.join(drive)+'''
enable=0;repeat(3)@(negedge clk);
if(addresses==0)$fatal(1,"Sparse address metadata was disabled");
$display("RETAINED_METADATA_PASS");$finish;end
endmodule
''')
    binary=tmp_path/'retained.vvp'
    subprocess.run(['iverilog','-g2012','-s','tb','-o',str(binary),str(tb),str(hdl/'data_pipeline_iact.v')],check=True,capture_output=True)
    log=subprocess.check_output(['vvp',str(binary)],text=True)
    actual=[int(line.split()[1])for line in log.splitlines()if line.startswith('VALUE ')]
    assert actual==[(i<<8)|v for i,v in enumerate(values)]
    assert 'RETAINED_METADATA_PASS'in log


def test_duplicate_or_unknown_decoder_cannot_be_silently_patched(tmp_path):
    (tmp_path/'hdl').mkdir();p=tmp_path/'hdl/data_pipeline_iact.v';p.write_text('unknown')
    with pytest.raises(ValueError):apply(tmp_path)
    assert p.read_text()=='unknown'
