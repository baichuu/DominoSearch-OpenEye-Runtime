"""Real Icarus regression for retained-zero address-count lifetime on pin363."""
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
PIN = '36395ddc530e106f06c50025a4069dd097688d8d'

BENCH = r'''
module tb;
  reg clk=0; always #5 clk=~clk;
  reg rst=0, enable=0, compute=0;
  reg [23:0] data=0;
  wire [4:0] count1, count2;
  wire en1;
  data_pipeline_iact #(.DATA_WIDTH(24), .FIRST_SPAD_ADDR(16),
    .FIRST_SPAD_DATA(4), .SECOND_SPAD_ADDR(16), .SECOND_SPAD_DATA(12),
    .SECOND_PAYLOAD_WIDTH(8), .SPARSITY_EN(0)) dut(
    .clk_i(clk),.rst_ni(rst),.compute_i(compute),.enable_i(enable),.data_i(data),
    .iact_x_line_repetitions_i(4'd1),.first_spad_max_i(4'd4),
    .first_spad_words_o(count1),.second_spad_words_o(count2),.first_spad_en_o(en1));
  initial begin
    repeat(3) @(negedge clk); rst=1;
    @(negedge clk); enable=1; data=24'h009007;
    repeat(2) @(negedge clk); data=0;
    repeat(2) @(negedge clk); enable=0;
    @(negedge clk);
    if (count1 !== 1 || count2 !== 4 || en1 !== 1)
      $fatal(1,"Retained-zero count was erased: first=%d second=%d enable=%d",count1,count2,en1);
    repeat(3) @(negedge clk);
    if (count1 !== 1 || count2 !== 4)
      $fatal(1,"Counts did not survive until compute");
    compute=1; @(negedge clk); compute=0;
    if (count1 !== 0) $fatal(1,"Count was not cleared at compute boundary");
    $display("RETAINED_ZERO_COUNT_LIFETIME_PASS"); $finish;
  end
endmodule
'''


def test_projection_count_survives_load_and_clears_only_at_compute(tmp_path):
    checkout = ROOT.parent / 'OpenEye'
    if not shutil.which('iverilog') or not shutil.which('vvp') or not checkout.exists():
        pytest.skip('Pinned checkout and Icarus required for this RTL regression')
    assert subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip() == PIN
    assert not subprocess.check_output(['git', '-C', str(checkout), 'status', '--short'], text=True).strip()
    upstream = subprocess.check_output(['git', '-C', str(checkout), 'show', PIN+':hdl/data_pipeline_iact.v'])
    (tmp_path/'tb.v').write_text(BENCH)
    results = []
    for label in ('baseline', 'fixed'):
        work = tmp_path/label; (work/'hdl').mkdir(parents=True)
        (work/'hdl/data_pipeline_iact.v').write_bytes(upstream)
        patches = ['openeye-activation-packing-compat.patch']
        if label == 'fixed':
            patches.append('openeye-runtime-projection-dense-iact.patch')
        for patch in patches:
            subprocess.run(['git', 'apply', '--include=hdl/data_pipeline_iact.v',
                str(ROOT/'fpga/openeye/patches'/patch)], cwd=work, check=True, capture_output=True)
        binary = work/'simulation'
        subprocess.run(['iverilog', '-g2012', '-s', 'tb', '-o', str(binary),
            str(tmp_path/'tb.v'), str(work/'hdl/data_pipeline_iact.v')], check=True, capture_output=True)
        results.append(subprocess.run(['vvp', str(binary)], capture_output=True, text=True, timeout=10))
    assert results[0].returncode != 0 and 'Retained-zero count was erased' in results[0].stdout
    assert results[1].returncode == 0 and 'RETAINED_ZERO_COUNT_LIFETIME_PASS' in results[1].stdout
