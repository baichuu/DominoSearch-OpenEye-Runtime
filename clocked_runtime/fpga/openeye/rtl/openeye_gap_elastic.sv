// Same nonnegative INT8 GAP arithmetic as openeye_gap_stream, with ready/valid.
// A stalled completed channel holds data and prevents accepting another input.
// pixels_i must be nonzero and constant until that channel's last input fires.
`timescale 1ns/1ps
module openeye_gap_elastic (
  input wire clk_i, rst_ni, valid_i,
  output wire ready_o,
  input wire [7:0] data_i,
  input wire [15:0] pixels_i,
  output reg valid_o,
  input wire ready_i,
  output reg [63:0] data_o
);
  reg [31:0] sum;
  reg [15:0] index;
  wire [31:0] total = sum + data_i;
  wire [31:0] quotient = pixels_i == 0 ? 0 : total / pixels_i;
  wire [31:0] remainder = pixels_i == 0 ? 0 : total % pixels_i;
  wire increment = (2*remainder > pixels_i) ||
                   ((2*remainder == pixels_i) && quotient[0]);
  assign ready_o = !valid_o || ready_i;
  always @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin sum<=0; index<=0; valid_o<=0; data_o<=0; end
    else begin
      if (ready_i) valid_o <= 0;
      if (valid_i && ready_o) begin
        if (index == pixels_i-1) begin
          sum <= 0; index <= 0; valid_o <= 1; data_o <= quotient + increment;
        end else begin sum <= total; index <= index + 1'b1; end
      end
    end
  end
endmodule
