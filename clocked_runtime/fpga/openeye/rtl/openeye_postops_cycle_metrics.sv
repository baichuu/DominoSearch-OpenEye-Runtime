// Observe actual register-valid boundaries of the existing fused pipeline.
// Tags classify affine/residual and ReLU/bypass; no operand enters the observer.
`timescale 1ns/1ps
module openeye_postops_cycle_metrics #(parameter integer SCOPE=0)(
  input wire clk_i,rst_ni,clear_i,dump_i,
  input wire [31:0] instruction_i,
  input wire valid_i,ready_i,residual_i,
  input wire valid_s1_i,valid_s2_i,valid_s3_i,relu_s2_i,core_valid_i,
  input wire output_valid_i,output_ready_i
);
  reg residual_s1,relu_s3;
  always @(posedge clk_i or negedge rst_ni)begin
    if(!rst_ni)begin residual_s1<=0;relu_s3<=0;end
    else begin residual_s1<=residual_i;relu_s3<=relu_s2_i;end
  end
  wire [6:0] start_valid,finish_valid,start_ready,finish_ready;
  assign start_valid={core_valid_i,valid_s3_i,valid_s2_i && !relu_s2_i,valid_s2_i && relu_s2_i,
                      valid_s1_i,valid_i && residual_i,valid_i && !residual_i};
  assign finish_valid={output_valid_i,core_valid_i,valid_s3_i && !relu_s3,valid_s3_i && relu_s3,
                       valid_s2_i,valid_s1_i && residual_s1,valid_s1_i && !residual_s1};
  assign start_ready={5'b11111,ready_i,ready_i};
  assign finish_ready={output_ready_i,6'b111111};
  generate for(genvar step=0;step<7;step=step+1)begin:functions
    openeye_function_cycle_observer #(.SCOPE(SCOPE),.FUNCTION_ID(step+1),.EXPECTED_LATENCY(step==6 ? -1 : 1)) observer(
      .clk_i(clk_i),.rst_ni(rst_ni),.clear_i(clear_i),.dump_i(dump_i),.instruction_i(instruction_i),
      .input_valid_i(start_valid[step]),.input_ready_i(start_ready[step]),
      .output_valid_i(finish_valid[step]),.output_ready_i(finish_ready[step]));
  end endgenerate
endmodule
