// Connected signed-INT8 Conv7x7 -> fixed BN/requant/ReLU -> maxpool3 frontend.
// Geometry is parameterized for focused verification; production geometry is
// C3x224x224 -> F64x56x56. The scalar stem is shared by all three experiments.
`timescale 1ns/1ps
module openeye_stem_frontend #(
  parameter integer WIDTH=224,HEIGHT=224,CHANNELS=3,FILTERS=64,KERNEL=7,STRIDE=2,
  parameter integer SHIFT=24,
  parameter integer CHANNEL_LANES=1,
  localparam integer CONV_WIDTH=(WIDTH+STRIDE-1)/STRIDE,
  localparam integer CONV_HEIGHT=(HEIGHT+STRIDE-1)/STRIDE
) (
  input wire clk_i,rst_ni,
  input wire image_write_i,weight_write_i,
  input wire [31:0] write_address_i,
  input wire signed [7:0] write_data_i,
  input wire coefficient_write_i,
  input wire [31:0] coefficient_address_i,
  input wire signed [63:0] multiplier_i,bias_i,
  input wire start_i,
  output wire command_ready_o,
  output wire valid_o,
  input wire ready_i,
  output wire [7:0] data_o,
  output wire done_o,invalid_o
);
  reg signed [63:0] multipliers[0:FILTERS-1],biases[0:FILTERS-1];
  reg configuration_invalid;
  wire conv_command_ready,pool_command_ready,conv_valid,conv_ready;
  wire conv_done,conv_invalid,pool_invalid;
  wire signed [31:0] conv_data;
  wire [31:0] conv_filter,conv_x,conv_y;
  wire post_valid,post_ready;
  wire signed [63:0] post_data;
  assign command_ready_o=conv_command_ready && pool_command_ready;
  wire start_fire=start_i && command_ready_o;
  assign invalid_o=configuration_invalid || conv_invalid || pool_invalid;
  always @(posedge clk_i or negedge rst_ni) begin
    if(!rst_ni)configuration_invalid<=0;
    else begin
      if(start_i && !command_ready_o)configuration_invalid<=1;
      if(coefficient_write_i)begin
        if(!command_ready_o || coefficient_address_i>=FILTERS)configuration_invalid<=1;
        else begin
          multipliers[coefficient_address_i]<=multiplier_i;
          biases[coefficient_address_i]<=bias_i;
        end
      end
    end
  end
  generate if(CHANNEL_LANES==1)begin:scalar
  openeye_stem_conv_stream #(.WIDTH(WIDTH),.HEIGHT(HEIGHT),.CHANNELS(CHANNELS),
    .FILTERS(FILTERS),.KERNEL(KERNEL),.STRIDE(STRIDE)) conv (
    .clk_i(clk_i),.rst_ni(rst_ni),.image_write_i(image_write_i),.weight_write_i(weight_write_i),
    .write_address_i(write_address_i),.write_data_i(write_data_i),.start_i(start_fire),
    .command_ready_o(conv_command_ready),.valid_o(conv_valid),.ready_i(conv_ready),.data_o(conv_data),
    .filter_o(conv_filter),.x_o(conv_x),.y_o(conv_y),.done_o(conv_done),.invalid_o(conv_invalid));
  end else begin:channel_parallel
  openeye_stem_conv_channel_parallel #(.WIDTH(WIDTH),.HEIGHT(HEIGHT),.CHANNELS(CHANNELS),
    .FILTERS(FILTERS),.KERNEL(KERNEL),.STRIDE(STRIDE),.CHANNEL_LANES(CHANNEL_LANES)) conv (
    .clk_i(clk_i),.rst_ni(rst_ni),.image_write_i(image_write_i),.weight_write_i(weight_write_i),
    .write_address_i(write_address_i),.write_data_i(write_data_i),.start_i(start_fire),
    .command_ready_o(conv_command_ready),.valid_o(conv_valid),.ready_i(conv_ready),.data_o(conv_data),
    .filter_o(conv_filter),.x_o(conv_x),.y_o(conv_y),.done_o(conv_done),.invalid_o(conv_invalid));
  end endgenerate
  openeye_postops_elastic post (
    .clk_i(clk_i),.rst_ni(rst_ni),.valid_i(conv_valid),.ready_o(conv_ready),
    .residual_i(1'b0),.relu_i(1'b1),.value_i({{32{conv_data[31]}},conv_data}),
    .skip_i(64'sd0),.multiplier_i(multipliers[conv_filter]),.bias_i(biases[conv_filter]),
    .skip_multiplier_i(64'sd0),.shift_i(6'(SHIFT)),
    .valid_o(post_valid),.ready_i(post_ready),.data_o(post_data));
  // The stream is FXY; one line spans Y, hence width=CONV_HEIGHT.
  openeye_maxpool3_stream #(.LINE_CAPACITY(CONV_HEIGHT+2)) pool (
    .clk_i(clk_i),.rst_ni(rst_ni),.start_i(start_fire),.command_ready_o(pool_command_ready),
    .width_i(16'(CONV_HEIGHT)),.height_i(16'(CONV_WIDTH)),.channels_i(16'(FILTERS)),
    .valid_i(post_valid),.ready_o(post_ready),.data_i(post_data[7:0]),
    .valid_o(valid_o),.ready_i(ready_i),.data_o(data_o),.done_o(done_o),.invalid_o(pool_invalid));
endmodule
