// Reserve storage for a WHOLE bounded raw-output frame before issuing a command.
// The OpenEye DMA output currently fails a direct stalled-sink control. During
// capture this adapter keeps ready high for the declared frame; the downstream
// may then stall freely while reading the captured words. No overflow/drop.
// Capacity must cover all raw words, including padding; unsupported commands
// fail closed. Large layers need smaller tiles or external-memory buffering.
`timescale 1ns/1ps
module openeye_dma_output_frame_buffer #(
  parameter integer CAPACITY=1024,
  localparam integer ADDR_WIDTH=CAPACITY<2 ? 1 : $clog2(CAPACITY)
) (
  input wire clk_i,rst_ni,start_i,
  output wire command_ready_o,
  input wire [31:0] frame_words_i,
  input wire valid_i,
  output wire ready_o,
  input wire [63:0] data_i,
  output wire valid_o,
  input wire ready_i,
  output wire [63:0] data_o,
  output reg done_o,invalid_o
);
  reg [63:0] memory[0:CAPACITY-1];
  reg capturing,draining;
  reg [31:0] words,write_index,read_index;
  assign command_ready_o=!capturing && !draining;
  assign ready_o=capturing;
  assign valid_o=draining;
  assign data_o=draining ? memory[read_index[ADDR_WIDTH-1:0]] : 64'd0;
  always @(posedge clk_i or negedge rst_ni)begin
    if(!rst_ni)begin capturing<=0;draining<=0;words<=0;write_index<=0;read_index<=0;done_o<=0;invalid_o<=0;end
    else begin
      done_o<=0;
      if(start_i && command_ready_o)begin
        if(frame_words_i==0 || frame_words_i>CAPACITY)invalid_o<=1;
        else begin capturing<=1;words<=frame_words_i;write_index<=0;read_index<=0;end
      end else if(start_i)invalid_o<=1;
      if(valid_i && ready_o)begin
        memory[write_index[ADDR_WIDTH-1:0]]<=data_i;
        if(write_index==words-1)begin capturing<=0;draining<=1;end
        else write_index<=write_index+1'b1;
      end
      if(valid_o && ready_i)begin
        if(read_index==words-1)begin draining<=0;done_o<=1;end
        else read_index<=read_index+1'b1;
      end
    end
  end
endmodule
