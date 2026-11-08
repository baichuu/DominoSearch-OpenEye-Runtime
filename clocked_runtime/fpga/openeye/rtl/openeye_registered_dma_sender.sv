// Conventional ready/valid upstream -> OpenEye's registered DMA input.
// OpenEye consumes last cycle's external valid/data with THIS cycle's ready.
// A one-cycle pulse followed by an acknowledgement edge avoids a duplicate
// tail word when ready falls at GET_BIAS/GET_QUANTIZE boundaries. A rejected
// pulse is retried with an empty pipeline slot. Conservative one word/3clocks.
// Reset must be released only after the accelerator's synchronized reset.
`timescale 1ns/1ps
module openeye_registered_dma_sender (
  input wire clk_i,rst_ni,
  input wire valid_i,
  output wire ready_o,
  input wire [63:0] data_i,
  output wire enable_dma_o,
  output wire [63:0] data_dma_o,
  input wire ready_dma_i,
  output wire accepted_o,
  output wire pending_o
);
  localparam [1:0] EMPTY=0,PULSE=1,ACK=2;
  reg [1:0] state;
  reg [63:0] payload;
  assign ready_o=state==EMPTY;
  assign enable_dma_o=state==PULSE;
  assign data_dma_o=payload;
  assign accepted_o=state==ACK && ready_dma_i;
  assign pending_o=state!=EMPTY;
  always @(posedge clk_i or negedge rst_ni)begin
    if(!rst_ni)begin state<=EMPTY;payload<=0;end
    else case(state)
      EMPTY: if(valid_i)begin payload<=data_i;state<=PULSE;end
      PULSE: state<=ACK;
      ACK: state<=ready_dma_i ? EMPTY : PULSE;
      default: state<=EMPTY;
    endcase
  end
endmodule
