// Research stem: channel-banked parallel signed INT8 MACs.
// One independent image/weight read per lane. The arithmetic and output order
// match the scalar engine; no timing/area claim until synthesis. Image CXY, weights C,F,KY,KX, output FXY.
// Write weights/model once; load each image through the write interface before
// start. The network-level interval must include that per-image loading time.
`timescale 1ns/1ps
module openeye_stem_conv_channel_parallel #(
  parameter integer WIDTH=224,HEIGHT=224,CHANNELS=3,FILTERS=64,KERNEL=7,STRIDE=2,
  parameter integer CHANNEL_LANES=3,
  localparam integer CHANNEL_GROUPS=CHANNELS/CHANNEL_LANES,
  localparam integer OUTPUT_WIDTH=(WIDTH+STRIDE-1)/STRIDE,
  localparam integer OUTPUT_HEIGHT=(HEIGHT+STRIDE-1)/STRIDE
) (
  input wire clk_i,rst_ni,
  input wire image_write_i,weight_write_i,
  input wire [31:0] write_address_i,
  input wire signed [7:0] write_data_i,
  input wire start_i,
  output wire command_ready_o,
  output reg valid_o,
  input wire ready_i,
  output reg signed [31:0] data_o,
  output reg [31:0] filter_o,x_o,y_o,
  output reg done_o,invalid_o
);
  reg active,draining;
  reg [31:0] f,x,y,c,kx,ky;
  reg signed [31:0] accumulator;
  wire signed [31:0] input_x=$signed(x*STRIDE+kx)-KERNEL/2;
  wire signed [31:0] input_y=$signed(y*STRIDE+ky)-KERNEL/2;
  wire outside=input_x<0 || input_y<0 || input_x>=WIDTH || input_y>=HEIGHT;
  wire [31:0] image_address=c*WIDTH*HEIGHT+input_x*HEIGHT+input_y;
  wire [31:0] weight_address=((c*FILTERS+f)*KERNEL+ky)*KERNEL+kx;
  localparam integer IMAGE_PLANE=WIDTH*HEIGHT,WEIGHT_PLANE=FILTERS*KERNEL*KERNEL;
  wire signed [15:0] products[0:CHANNEL_LANES-1];
  // A bank contains channels lane, lane+CHANNEL_LANES, ... . Total storage
  // bits do not grow with lanes; independent banks supply parallel read ports.
  generate for(genvar lane=0;lane<CHANNEL_LANES;lane=lane+1)begin:banks
    reg signed [7:0] image[0:CHANNEL_GROUPS*IMAGE_PLANE-1];
    reg signed [7:0] weights[0:CHANNEL_GROUPS*WEIGHT_PLANE-1];
    wire signed [7:0] activation=outside ? 8'sd0 : image[image_address];
    assign products[lane]=activation*weights[weight_address];
    always @(posedge clk_i)if(rst_ni && !active)begin
      if(image_write_i && write_address_i<CHANNELS*IMAGE_PLANE &&
         (write_address_i/IMAGE_PLANE)%CHANNEL_LANES==lane)
        image[(write_address_i/IMAGE_PLANE/CHANNEL_LANES)*IMAGE_PLANE+write_address_i%IMAGE_PLANE]<=write_data_i;
      if(weight_write_i && write_address_i<CHANNELS*WEIGHT_PLANE &&
         (write_address_i/WEIGHT_PLANE)%CHANNEL_LANES==lane)
        weights[(write_address_i/WEIGHT_PLANE/CHANNEL_LANES)*WEIGHT_PLANE+write_address_i%WEIGHT_PLANE]<=write_data_i;
    end
  end endgenerate
  reg signed [32:0] sum;
  integer lane_index;
  always @*begin
    sum={accumulator[31],accumulator};
    for(lane_index=0;lane_index<CHANNEL_LANES;lane_index=lane_index+1)
      sum=sum+{{17{products[lane_index][15]}},products[lane_index]};
  end
  initial begin
    if((CHANNEL_LANES!=1 && CHANNEL_LANES!=3) || CHANNELS%CHANNEL_LANES!=0)
      $fatal(1,"Unsupported stem channel banking");
  end
  assign command_ready_o=!active && !valid_o;
  always @(posedge clk_i or negedge rst_ni)begin
    if(!rst_ni)begin
      active<=0;draining<=0;f<=0;x<=0;y<=0;c<=0;kx<=0;ky<=0;accumulator<=0;
      valid_o<=0;data_o<=0;filter_o<=0;x_o<=0;y_o<=0;done_o<=0;invalid_o<=0;
    end else begin
      done_o<=0;
      if(image_write_i || weight_write_i)begin
        if(active)invalid_o<=1;
        else begin
          if(image_write_i && write_address_i>=CHANNELS*WIDTH*HEIGHT)invalid_o<=1;
          if(weight_write_i && write_address_i>=CHANNELS*FILTERS*KERNEL*KERNEL)invalid_o<=1;
        end
      end
      if(start_i && command_ready_o)begin
        active<=1;draining<=0;f<=0;x<=0;y<=0;c<=0;kx<=0;ky<=0;accumulator<=0;
      end else if(start_i)invalid_o<=1;
      if(valid_o && ready_i)begin
        valid_o<=0;
        if(draining)begin active<=0;draining<=0;done_o<=1;end
      end
      if(active && !draining && (!valid_o || ready_i))begin
        if(sum[32]!=sum[31])invalid_o<=1;
        accumulator<=sum[31:0];
        if(kx==KERNEL-1)begin
          kx<=0;
          if(ky==KERNEL-1)begin
            ky<=0;
            if(c==CHANNEL_GROUPS-1)begin
              c<=0;accumulator<=0;valid_o<=1;data_o<=sum[31:0];filter_o<=f;x_o<=x;y_o<=y;
              if(y==OUTPUT_HEIGHT-1)begin
                y<=0;
                if(x==OUTPUT_WIDTH-1)begin
                  x<=0;
                  if(f==FILTERS-1)draining<=1;
                  else f<=f+1'b1;
                end else x<=x+1'b1;
              end else y<=y+1'b1;
            end else c<=c+1'b1;
          end else ky<=ky+1'b1;
        end else kx<=kx+1'b1;
      end
    end
  end
endmodule
