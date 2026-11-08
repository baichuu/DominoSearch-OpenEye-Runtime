// Fixed-point adapter after a complete, channel-accumulated OpenEye PSUM.
// Matches Domino runtime arithmetic: int64 affine/alignment, ties-to-even,
// signed [-127,127] saturation and optional signed ReLU. Four pipeline edges:
// affine/alignment -> round -> pinned af_cluster -> saturation. No backpressure
// in this bridge: caller must provide an always-ready sink. All coefficients
// are supplied per value; range/overflow is checked before ingress by the host.
`timescale 1ns/1ps
module openeye_postops_stream (
  input wire clk_i, rst_ni, valid_i,
  input wire residual_i, relu_i,
  input wire signed [63:0] value_i, skip_i, multiplier_i, bias_i, skip_multiplier_i,
  input wire [5:0] shift_i,
  output reg valid_o,
  output reg signed [63:0] data_o
);
  reg signed [63:0] wide_s1, round_s2;
  reg [5:0] shift_s1;
  reg relu_s1, relu_s2;
  reg valid_s1, valid_s2, valid_s3;
  wire [63:0] activation;
  reg signed [63:0] quotient;
  reg [63:0] mask, remainder, half;
  always @* begin
    quotient=wide_s1 >>> shift_s1;
    mask=(64'd1 << shift_s1)-1;
    remainder=$unsigned(wide_s1)&mask;
    half=64'd1 << (shift_s1-1);
    if (shift_s1!=0 && (remainder>half || (remainder==half && quotient[0])))
      quotient=quotient+1;
  end
  af_cluster #(.WIDTH(64),.DEPTH(1),.VECTOR_SIZE(1)) activation_stage (
    .clk_i(clk_i),.rst_ni(rst_ni),.relu_en_i(relu_s2),.data_i(round_s2),.data_o(activation));
  always @(posedge clk_i or negedge rst_ni) begin
    if(!rst_ni) begin
      wide_s1<=0;round_s2<=0;shift_s1<=0;relu_s1<=0;relu_s2<=0;
      valid_s1<=0;valid_s2<=0;valid_s3<=0;valid_o<=0;data_o<=0;
    end else begin
      valid_s1<=valid_i;valid_s2<=valid_s1;valid_s3<=valid_s2;valid_o<=valid_s3;
      shift_s1<=shift_i;relu_s1<=relu_i;relu_s2<=relu_s1;
      if(residual_i) wide_s1<=value_i*multiplier_i+skip_i*skip_multiplier_i;
      else wide_s1<=value_i*multiplier_i+bias_i;
      round_s2<=quotient;
      if($signed(activation)>64'sd127) data_o<=127;
      else if($signed(activation)<-64'sd127) data_o<=-127;
      else data_o<=activation;
    end
  end
endmodule
