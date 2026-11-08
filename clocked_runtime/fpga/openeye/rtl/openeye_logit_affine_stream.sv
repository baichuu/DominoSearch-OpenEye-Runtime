// INT32 FC PSUM -> signed INT64 Q40 logit. No INT8 clamp or ReLU at logits.
// Coefficients are frozen model setup; m=round(scale*2^40), b=round(bias*2^40).
`timescale 1ns/1ps
module openeye_logit_affine_stream (
  input wire clk_i,rst_ni,valid_i,
  output wire ready_o,
  input wire signed [31:0] data_i,
  input wire signed [63:0] multiplier_i,bias_i,
  output reg valid_o,
  input wire ready_i,
  output reg signed [63:0] data_o,
  output reg invalid_o
);
  wire signed [95:0] product=data_i*multiplier_i;
  wire signed [95:0] sum=product+{{32{bias_i[63]}},bias_i};
  assign ready_o=!valid_o || ready_i;
  always @(posedge clk_i or negedge rst_ni)begin
    if(!rst_ni)begin valid_o<=0;data_o<=0;invalid_o<=0;end
    else begin
      if(valid_o && ready_i)valid_o<=0;
      if(valid_i && ready_o)begin
        if(sum[95:63]!={33{sum[63]}})invalid_o<=1;
        valid_o<=1;data_o<=sum[63:0];
      end
    end
  end
endmodule
