// Nonnegative INT8 maxpool3, stride2, SAME padding. One channel plane at a
// time; the two line buffers are reused across channels. Padding is generated
// in RTL and consumes clock cycles. All output stalls propagate to the input.
`timescale 1ns/1ps
module openeye_maxpool3_stream #(
  parameter integer LINE_CAPACITY=226
) (
  input wire clk_i,rst_ni,start_i,
  output wire command_ready_o,
  input wire [15:0] width_i,height_i,channels_i,
  input wire valid_i,
  output wire ready_o,
  input wire [7:0] data_i,
  output reg valid_o,
  input wire ready_i,
  output reg [7:0] data_o,
  output reg done_o,invalid_o
);
  reg active;
  reg [15:0] width,height,channels,row,col,channel;
  reg [7:0] line1[0:LINE_CAPACITY-1],line2[0:LINE_CAPACITY-1];
  reg [7:0] top_left,top_mid,middle_left,middle_mid,bottom_left,bottom_mid;
  wire padding_pixel=row==0 || col==0 || row==height+1 || col==width+1;
  wire can_step=active && (!valid_o || ready_i);
  wire [7:0] pixel=padding_pixel ? 8'd0 : data_i;
  wire [7:0] top=line2[col],middle=line1[col];
  reg [7:0] maximum;
  assign command_ready_o=!active && !valid_o;
  assign ready_o=can_step && !padding_pixel;
  always @* begin
    maximum=top_left;
    if(top_mid>maximum)maximum=top_mid;
    if(top>maximum)maximum=top;
    if(middle_left>maximum)maximum=middle_left;
    if(middle_mid>maximum)maximum=middle_mid;
    if(middle>maximum)maximum=middle;
    if(bottom_left>maximum)maximum=bottom_left;
    if(bottom_mid>maximum)maximum=bottom_mid;
    if(pixel>maximum)maximum=pixel;
  end
  always @(posedge clk_i or negedge rst_ni) begin
    if(!rst_ni)begin
      active<=0;width<=0;height<=0;channels<=0;row<=0;col<=0;channel<=0;
      top_left<=0;top_mid<=0;middle_left<=0;middle_mid<=0;bottom_left<=0;bottom_mid<=0;
      valid_o<=0;data_o<=0;done_o<=0;invalid_o<=0;
    end else begin
      done_o<=0;
      if(valid_o && ready_i)valid_o<=0;
      if(start_i && command_ready_o)begin
        if(width_i==0 || height_i==0 || channels_i==0 || width_i>LINE_CAPACITY-2 || height_i>65533)invalid_o<=1;
        else begin
          active<=1;width<=width_i;height<=height_i;channels<=channels_i;
          row<=0;col<=0;channel<=0;
        end
      end else if(start_i)invalid_o<=1;
      if(can_step && (padding_pixel || valid_i))begin
        line2[col]<=middle;line1[col]<=pixel;
        top_left<=col==0 ? 0 : top_mid;top_mid<=top;
        middle_left<=col==0 ? 0 : middle_mid;middle_mid<=middle;
        bottom_left<=col==0 ? 0 : bottom_mid;bottom_mid<=pixel;
        if(row>=2 && col>=2 && row[0]==0 && col[0]==0)begin
          valid_o<=1;data_o<=maximum;
        end
        if(col==width+1)begin
          col<=0;
          if(row==height+1)begin
            row<=0;
            if(channel==channels-1)begin active<=0;done_o<=1;end
            else channel<=channel+1'b1;
          end else row<=row+1'b1;
        end else col<=col+1'b1;
      end
    end
  end
endmodule
