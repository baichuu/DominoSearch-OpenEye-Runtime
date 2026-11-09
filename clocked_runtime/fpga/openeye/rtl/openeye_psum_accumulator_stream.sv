// Accumulate complete, identically ordered channel-group partials in INT32.
// A command supplies groups_i vectors of elements_i values. First group writes
// every address before any read; only the last group emits final PSUMs.
// Spatial/filter order restoration belongs to the upstream address mapper.
`timescale 1ns/1ps
module openeye_psum_accumulator_stream #(
  parameter integer CAPACITY=4096,
  localparam integer ADDR_WIDTH=CAPACITY<2 ? 1 : $clog2(CAPACITY)
) (
  input wire clk_i,rst_ni,
  input wire start_i,
  output wire command_ready_o,
  input wire [31:0] elements_i,groups_i,
  input wire valid_i,
  output wire ready_o,
  input wire signed [31:0] data_i,
  output reg valid_o,
  input wire ready_i,
  output reg signed [31:0] data_o,
  output reg [31:0] index_o,
  output reg done_o,invalid_o
);
  reg signed [31:0] memory[0:CAPACITY-1];
  reg active,draining;
  reg [31:0] elements,groups,index,group_index;
  wire [ADDR_WIDTH-1:0] address=index[ADDR_WIDTH-1:0];
  wire signed [32:0] extended_input={data_i[31],data_i};
  wire signed [32:0] extended_previous={memory[address][31],memory[address]};
  wire signed [32:0] sum=extended_input+(group_index==0 ? 33'sd0 : extended_previous);
  assign command_ready_o=!active;
  assign ready_o=active && !draining && (!valid_o || ready_i);
  always @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      active<=0;draining<=0;elements<=0;groups<=0;index<=0;group_index<=0;
      valid_o<=0;data_o<=0;index_o<=0;done_o<=0;invalid_o<=0;
    end else begin
      done_o<=0;
      if (start_i && command_ready_o) begin
        if (elements_i==0 || elements_i>CAPACITY || groups_i==0) invalid_o<=1;
        else begin
          active<=1;draining<=0;elements<=elements_i;groups<=groups_i;
          index<=0;group_index<=0;valid_o<=0;
        end
      end else if (start_i) invalid_o<=1;
      if (valid_o && ready_i) begin
        valid_o<=0;
        if (draining) begin active<=0;draining<=0;done_o<=1;end
      end
      if (valid_i && ready_o) begin
        if (sum[32]!=sum[31]) invalid_o<=1;
        memory[address]<=sum[31:0];
        if (group_index==groups-1) begin
          valid_o<=1;data_o<=sum[31:0];index_o<=index;
        end
        if (index==elements-1) begin
          index<=0;
          if (group_index==groups-1) draining<=1;
          else group_index<=group_index+1'b1;
        end else index<=index+1'b1;
      end
    end
  end
endmodule
