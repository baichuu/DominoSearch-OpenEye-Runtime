// Signed INT8 scalar stem engine, independent of the residual OpenEye core.
// Common to dense/MAC20/optimized experiments. One MAC per active clock; no
// timing/area claim until synthesis. Image CXY, weights C,F,KY,KX, output FXY.
// Write weights/model once; load each image through the write interface before
// start. The network-level interval must include that per-image loading time.
`timescale 1ns/1ps
module openeye_stem_conv_stream #(
  parameter integer WIDTH=224,HEIGHT=224,CHANNELS=3,FILTERS=64,KERNEL=7,STRIDE=2,
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
  reg signed [7:0] image[0:CHANNELS*WIDTH*HEIGHT-1];
  reg signed [7:0] weights[0:CHANNELS*FILTERS*KERNEL*KERNEL-1];
  reg active,draining;
  reg [31:0] f,x,y,c,kx,ky;
  reg signed [31:0] accumulator;
  wire signed [31:0] input_x=$signed(x*STRIDE+kx)-KERNEL/2;
  wire signed [31:0] input_y=$signed(y*STRIDE+ky)-KERNEL/2;
  wire outside=input_x<0 || input_y<0 || input_x>=WIDTH || input_y>=HEIGHT;
  wire [31:0] image_address=c*WIDTH*HEIGHT+input_x*HEIGHT+input_y;
  wire [31:0] weight_address=((c*FILTERS+f)*KERNEL+ky)*KERNEL+kx;
  wire signed [7:0] activation=outside ? 8'sd0 : image[image_address];
  wire signed [15:0] product=activation*weights[weight_address];
  wire signed [32:0] sum={accumulator[31],accumulator}+{{17{product[15]}},product};
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
          if(image_write_i)begin
            if(write_address_i<CHANNELS*WIDTH*HEIGHT)image[write_address_i]<=write_data_i;
            else invalid_o<=1;
          end
          if(weight_write_i)begin
            if(write_address_i<CHANNELS*FILTERS*KERNEL*KERNEL)weights[write_address_i]<=write_data_i;
            else invalid_o<=1;
          end
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
            if(c==CHANNELS-1)begin
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
