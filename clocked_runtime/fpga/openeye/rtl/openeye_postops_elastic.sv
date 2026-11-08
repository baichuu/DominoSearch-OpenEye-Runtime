// Credit-buffered adapter for the fixed four-stage post-op pipeline.
// Reserve one FIFO slot before accepting each input, including values still
// in flight. The OpenEye af_cluster stage can then keep running during stalls.
// No gated clock and no dropped/duplicated PSUMs on output backpressure.
`timescale 1ns/1ps
module openeye_postops_elastic #(
  parameter integer FIFO_DEPTH = 8,
  localparam integer PTR_WIDTH = FIFO_DEPTH < 2 ? 1 : $clog2(FIFO_DEPTH),
  localparam integer COUNT_WIDTH = $clog2(FIFO_DEPTH+1)
) (
  input wire clk_i, rst_ni,
  input wire valid_i,
  output wire ready_o,
  input wire residual_i, relu_i,
  input wire signed [63:0] value_i, skip_i, multiplier_i, bias_i, skip_multiplier_i,
  input wire [5:0] shift_i,
  output wire valid_o,
  input wire ready_i,
  output wire signed [63:0] data_o
);
  reg [COUNT_WIDTH-1:0] reserved, queued;
  reg [PTR_WIDTH-1:0] read_ptr, write_ptr;
  reg signed [63:0] fifo [0:FIFO_DEPTH-1];
  wire core_valid;
  wire signed [63:0] core_data;
  wire output_fire = valid_o && ready_i;
  wire input_fire = valid_i && ready_o;
  assign valid_o = queued != 0;
  assign ready_o = reserved < FIFO_DEPTH || output_fire;
  assign data_o = valid_o ? fifo[read_ptr] : 64'sd0;

  openeye_postops_stream core (
    .clk_i(clk_i),.rst_ni(rst_ni),.valid_i(input_fire),
    .residual_i(residual_i),.relu_i(relu_i),
    .value_i(value_i),.skip_i(skip_i),.multiplier_i(multiplier_i),
    .bias_i(bias_i),.skip_multiplier_i(skip_multiplier_i),.shift_i(shift_i),
    .valid_o(core_valid),.data_o(core_data));

  always @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      reserved <= 0; queued <= 0; read_ptr <= 0; write_ptr <= 0;
    end else begin
      case ({input_fire,output_fire})
        2'b10: reserved <= reserved + 1'b1;
        2'b01: reserved <= reserved - 1'b1;
        default: begin end
      endcase
      case ({core_valid,output_fire})
        2'b10: queued <= queued + 1'b1;
        2'b01: queued <= queued - 1'b1;
        default: begin end
      endcase
      if (core_valid) begin
        fifo[write_ptr] <= core_data;
        write_ptr <= write_ptr == FIFO_DEPTH-1 ? 0 : write_ptr + 1'b1;
      end
      if (output_fire)
        read_ptr <= read_ptr == FIFO_DEPTH-1 ? 0 : read_ptr + 1'b1;
    end
  end
endmodule
