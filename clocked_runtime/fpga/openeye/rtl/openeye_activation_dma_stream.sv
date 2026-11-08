// Activation SRAM -> paired-channel OpenEye raw 64-bit DMA order.
// CXY INT8 SRAM: READ_LATENCY=0 combinational, READ_LATENCY=1 registered read.
// In registered mode issue one address, then consume its data at the next edge;
// padding never issues a read. Both modes include memory wait cycles in timeline.
// Channel slices are 64, used channels 4(K3)/8(K1). Stage eight complete words,
// then send them in reverse order as OpenEye's RAM-cell shift writer requires.
// Mapping: native/halo copy and far padding, or CXY raster after predecimation.
`timescale 1ns/1ps
module openeye_activation_dma_stream #(
  parameter integer READ_LATENCY=0
) (
  input wire clk_i,rst_ni,start_i,
  output wire command_ready_o,
  input wire [31:0] base_address_i,
  input wire [15:0] source_width_i,source_height_i,source_channels_i,
  input wire [15:0] channel_start_i,
  input wire [15:0] target_width_i,target_height_i,
  input wire [15:0] source_x_start_i,copy_width_i,copy_height_i,
  input wire raster_i,predecimate_i,
  input wire [3:0] used_channels_i,
  output wire read_enable_o,
  output wire [31:0] read_address_o,
  input wire [7:0] read_data_i,
  output wire valid_o,
  input wire ready_i,
  output wire [63:0] data_o,
  output reg done_o,invalid_o
);
  reg active,sending,final_group,read_pending;
  reg [31:0] base,total_bytes,processed;
  reg [15:0] source_width,source_height,channel_start,target_width,target_height;
  reg [15:0] source_x_start,copy_width,copy_height;
  reg raster,predecimate;
  reg [3:0] used_channels;
  reg [15:0] x,y;
  reg [6:0] channel_group,channel_in_group;
  reg [5:0] fill;
  reg [2:0] send_word;
  reg [511:0] staging;
  wire [31:0] logical_width=(source_width+(predecimate?1:0))/(predecimate?2:1);
  wire [31:0] logical_height=(source_height+(predecimate?1:0))/(predecimate?2:1);
  wire [31:0] raster_pixel=x*target_height+y;
  wire [31:0] mapped_x=raster ? (raster_pixel/logical_height)*(predecimate?2:1) : source_x_start+x;
  wire [31:0] mapped_y=raster ? (raster_pixel%logical_height)*(predecimate?2:1) : y;
  wire pad=raster ? raster_pixel>=logical_width*logical_height : x>=copy_width || y>=copy_height;
  assign command_ready_o=!active;
  assign read_enable_o=active && !sending && !pad && (READ_LATENCY==0 || !read_pending);
  assign read_address_o=base+(channel_start+channel_group+channel_in_group)*source_width*source_height+
      mapped_x*source_height+mapped_y;
  assign valid_o=active && sending;
  assign data_o=staging[send_word*64+:64];
  always @(posedge clk_i or negedge rst_ni)begin
    if(!rst_ni)begin
      active<=0;sending<=0;final_group<=0;read_pending<=0;base<=0;total_bytes<=0;processed<=0;
      source_width<=1;source_height<=1;channel_start<=0;target_width<=0;target_height<=0;
      source_x_start<=0;copy_width<=0;copy_height<=0;raster<=0;predecimate<=0;used_channels<=4;
      x<=0;y<=0;channel_group<=0;channel_in_group<=0;fill<=0;send_word<=7;staging<=0;
      done_o<=0;invalid_o<=0;
    end else begin
      done_o<=0;
      if(start_i && command_ready_o)begin
        if((READ_LATENCY!=0 && READ_LATENCY!=1) ||
           source_width_i==0 || source_height_i==0 || target_width_i==0 || target_height_i==0 ||
           channel_start_i+32'd64>source_channels_i || (used_channels_i!=4 && used_channels_i!=8) ||
           target_width_i>256 || target_height_i>256 ||
           (!raster_i && (source_x_start_i+32'(copy_width_i)>source_width_i || copy_height_i>source_height_i)))invalid_o<=1;
        else begin
          active<=1;sending<=0;final_group<=0;read_pending<=0;staging<=0;fill<=0;send_word<=7;
          base<=base_address_i;total_bytes<=32'd64*target_width_i*target_height_i;processed<=0;
          source_width<=source_width_i;source_height<=source_height_i;channel_start<=channel_start_i;
          target_width<=target_width_i;target_height<=target_height_i;
          source_x_start<=source_x_start_i;copy_width<=copy_width_i;copy_height<=copy_height_i;
          raster<=raster_i;predecimate<=predecimate_i;used_channels<=used_channels_i;
          x<=0;y<=0;channel_group<=0;channel_in_group<=0;
        end
      end else if(start_i)invalid_o<=1;
      if(READ_LATENCY==1 && read_enable_o)read_pending<=1;
      if(active && !sending && (pad || READ_LATENCY==0 || read_pending))begin
        read_pending<=0;
        staging[fill*8+:8]<=pad ? 8'd0 : read_data_i;
        processed<=processed+1;
        if(channel_in_group==used_channels-1)begin
          channel_in_group<=0;
          if(x==target_width-1)begin
            x<=0;
            if(y==target_height-1)begin y<=0;channel_group<=channel_group+used_channels;end
            else y<=y+1'b1;
          end else x<=x+1'b1;
        end else channel_in_group<=channel_in_group+1'b1;
        if(fill==63 || processed==total_bytes-1)begin
          fill<=0;sending<=1;send_word<=7;final_group<=processed==total_bytes-1;
        end else fill<=fill+1'b1;
      end
      if(valid_o && ready_i)begin
        if(send_word==0)begin
          send_word<=7;sending<=0;staging<=0;
          if(final_group)begin active<=0;done_o<=1;end
        end else send_word<=send_word-1'b1;
      end
    end
  end
endmodule
