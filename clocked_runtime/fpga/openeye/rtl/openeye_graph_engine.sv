// Clocked microprogram controller for the full-network experiment.
// No host tensor arithmetic: DMA activation fields read live scratch memory;
// raw OpenEye outputs scatter/add into INT32 PSUM memory; affine, residual,
// GAP and logits execute actual RTL. Model/program ROM setup is external.
// Scratch memory has one synchronous read and one write port. Mapping ROMs
// contain addresses/validity only, never golden tensor values.
`timescale 1ns/1ps
module openeye_graph_engine #(
  parameter integer FRAME_CAPACITY=65536
) (
  input wire clk_i,rst_ni,start_i,
  output wire command_ready_o,
  output reg [31:0] instruction_address_o,
  input wire [1023:0] instruction_i,
  output wire [31:0] packet_address_o,
  input wire [63:0] packet_data_i,
  output wire [31:0] map_address_o,
  input wire [31:0] map_data_i,
  output wire [31:0] coefficient_address_o,
  input wire signed [63:0] coefficient_multiplier_i,coefficient_bias_i,
  output reg memory_read_o,memory_write_o,
  output reg [31:0] memory_address_o,
  input wire signed [63:0] memory_data_i,
  output reg signed [63:0] memory_data_o,
  output wire enable_dma_o,
  output wire [63:0] data_dma_o,
  input wire ready_dma_i,core_valid_i,
  input wire [63:0] core_data_i,
  output wire core_ready_o,
  output reg done_o,invalid_o,
  output reg instruction_done_o,
  output reg [31:0] completed_instruction_o,
  output reg [31:0] accepted_words_o,raw_words_o,
  output reg [63:0] elapsed_cycles_o
);
  // Word fields:0op,1.. operation arguments. DMA uses1packetbase,2words,
  // 3iactoffset,4iactwords,5rawmapbase,6rawwords,7accumulatorbase,8groupindex,
  // 9..20 activation encoder descriptor,21finalgroupcount (audit metadata).
  // AFFINE:1src,2dst,3count,4coeffbase,5pixels,6relu,7shift.
  // RESIDUAL:1src,2dst,3count,4coeffbase,5unused,6skipbase,7shift.
  // GAP:1src,2dst,3count,4pixels. LOGIT:1src,2dst,3count,4coeffbase.
  localparam integer IDLE=0,FETCH=1,DMA_START=2,DMA_FEED=3,DMA_WAIT=4,
    SCATTER=5,ACC_READ=6,ACC_WRITE=7,SCAN_READ=8,SCAN_WAIT=9,
    SKIP_READ=10,SKIP_WAIT=11,POST_SEND=12,POST_WAIT=13,STORE=14,
    GAP_SEND=15,GAP_WAIT=16,LOGIT_SEND=17,LOGIT_WAIT=18,NEXT=19;
  reg [4:0] state;
  reg [1023:0] instruction;
  wire [31:0] op=instruction[0+:32];
  wire [31:0] a1=instruction[32+:32],a2=instruction[64+:32],a3=instruction[96+:32];
  wire [31:0] a4=instruction[128+:32],a5=instruction[160+:32],a6=instruction[192+:32],a7=instruction[224+:32],a8=instruction[256+:32];
  reg [31:0] sent,raw_index,index,output_index,lane;
  reg [63:0] raw_word;
  reg signed [63:0] value,skip_value,store_value;
  reg [31:0] destination;
  reg frame_start,encoder_start;
  wire frame_ready,frame_valid,frame_done,frame_invalid;
  wire [63:0] frame_data;
  wire source_ready,sender_pending,sender_accepted;
  wire encoder_ready,encoder_valid,encoder_done,encoder_invalid,encoder_read;
  wire [63:0] encoder_data;
  wire [31:0] encoder_address;
  wire in_activation=sent>=a3 && sent<a3+a4;
  wire linear_padding=instruction[320+:32]==1 && instruction[352+:32]==1 &&
      instruction[448+:32]==1 && instruction[480+:32]==1 && sent>=a3+8;
  wire feeding=state==DMA_FEED && sent<a2;
  wire source_valid=feeding && (!in_activation || linear_padding || encoder_valid);
  wire post_ready,post_valid;
  wire signed [63:0] post_data;
  wire gap_ready,gap_valid,logit_ready,logit_valid,logit_invalid;
  wire [63:0] gap_data;
  wire signed [63:0] logit_data;
  wire signed [31:0] partial=lane==0 ? $signed(raw_word[31:0]) : $signed(raw_word[63:32]);
  wire signed [64:0] accumulated=$signed({memory_data_i[63],memory_data_i})+{{33{partial[31]}},partial};
  assign command_ready_o=state==IDLE;
  assign packet_address_o=a1+sent;
  assign map_address_o=a5+raw_index*2+lane;
  assign coefficient_address_o=a4+(op==2 ? index/a5 : op==5 ? index : 0);
  openeye_registered_dma_sender sender (
    .clk_i(clk_i),.rst_ni(rst_ni),.valid_i(source_valid),.ready_o(source_ready),
    .data_i(in_activation ? (linear_padding ? 64'd0 : encoder_data) : packet_data_i),.enable_dma_o(enable_dma_o),.data_dma_o(data_dma_o),
    .ready_dma_i(ready_dma_i),.accepted_o(sender_accepted),.pending_o(sender_pending));
  openeye_dma_output_frame_buffer #(.CAPACITY(FRAME_CAPACITY)) frame (
    .clk_i(clk_i),.rst_ni(rst_ni),.start_i(frame_start),.command_ready_o(frame_ready),.frame_words_i(a6),
    .valid_i(core_valid_i),.ready_o(core_ready_o),.data_i(core_data_i),
    .valid_o(frame_valid),.ready_i(state==SCATTER && lane==0),.data_o(frame_data),.done_o(frame_done),.invalid_o(frame_invalid));
  openeye_activation_dma_stream #(.READ_LATENCY(1)) encoder (
    .clk_i(clk_i),.rst_ni(rst_ni),.start_i(encoder_start),.command_ready_o(encoder_ready),
    .base_address_i(instruction[288+:32]),.source_width_i(instruction[320+:16]),.source_height_i(instruction[352+:16]),
    .source_channels_i(instruction[384+:16]),.channel_start_i(instruction[416+:16]),
    .target_width_i(instruction[448+:16]),.target_height_i(instruction[480+:16]),
    .source_x_start_i(instruction[512+:16]),.copy_width_i(instruction[544+:16]),.copy_height_i(instruction[576+:16]),
    .raster_i(instruction[608]),.predecimate_i(instruction[640]),.used_channels_i(instruction[672+:4]),
    .read_enable_o(encoder_read),.read_address_o(encoder_address),.read_data_i(memory_data_i[7:0]),
    .valid_o(encoder_valid),.ready_i(feeding && in_activation && !linear_padding && source_ready),.data_o(encoder_data),
    .done_o(encoder_done),.invalid_o(encoder_invalid));
  openeye_postops_elastic post (
    .clk_i(clk_i),.rst_ni(rst_ni),.valid_i(state==POST_SEND),.ready_o(post_ready),
    .residual_i(op==3),.relu_i(op==3 || a6!=0),.value_i(value),.skip_i(skip_value),
    .multiplier_i(coefficient_multiplier_i),.bias_i(coefficient_bias_i),
    .skip_multiplier_i(coefficient_bias_i),.shift_i(a7[5:0]),
    .valid_o(post_valid),.ready_i(state==POST_WAIT),.data_o(post_data));
  openeye_gap_elastic gap (
    .clk_i(clk_i),.rst_ni(rst_ni),.valid_i(state==GAP_SEND),.ready_o(gap_ready),.data_i(value[7:0]),.pixels_i(a4[15:0]),
    .valid_o(gap_valid),.ready_i(state==GAP_WAIT),.data_o(gap_data));
  openeye_logit_affine_stream logit (
    .clk_i(clk_i),.rst_ni(rst_ni),.valid_i(state==LOGIT_SEND),.ready_o(logit_ready),.data_i(value[31:0]),
    .multiplier_i(coefficient_multiplier_i),.bias_i(coefficient_bias_i),
    .valid_o(logit_valid),.ready_i(state==LOGIT_WAIT),.data_o(logit_data),.invalid_o(logit_invalid));
  always @*begin
    memory_read_o=0;memory_write_o=0;memory_address_o=0;memory_data_o=store_value;
    if(state==DMA_FEED && encoder_read)begin memory_read_o=1;memory_address_o=encoder_address;end
    if(state==ACC_READ && map_data_i!=32'hffffffff)begin memory_read_o=1;memory_address_o=a7+map_data_i;end
    if(state==ACC_WRITE)begin memory_write_o=1;memory_address_o=a7+map_data_i;
      memory_data_o=a8==0 ? {{32{partial[31]}},partial} : accumulated[63:0];end
    if(state==SCAN_READ)begin memory_read_o=1;memory_address_o=a1+index;end
    if(state==SKIP_READ)begin memory_read_o=1;memory_address_o=a6+index;end
    if(state==STORE)begin memory_write_o=1;memory_address_o=a2+output_index;end
  end
  always @(posedge clk_i or negedge rst_ni)begin
    if(!rst_ni)begin
      state<=IDLE;instruction_address_o<=0;instruction<=0;done_o<=0;invalid_o<=0;instruction_done_o<=0;
      completed_instruction_o<=0;sent<=0;raw_index<=0;index<=0;output_index<=0;lane<=0;raw_word<=0;
      value<=0;skip_value<=0;store_value<=0;destination<=0;frame_start<=0;encoder_start<=0;
      accepted_words_o<=0;raw_words_o<=0;elapsed_cycles_o<=0;
    end else begin
      frame_start<=0;encoder_start<=0;done_o<=0;instruction_done_o<=0;
      if(state!=IDLE)elapsed_cycles_o<=elapsed_cycles_o+1'b1;
      if(frame_invalid || encoder_invalid || logit_invalid)invalid_o<=1;
      if(start_i && state!=IDLE)invalid_o<=1;
      if(sender_accepted)accepted_words_o<=accepted_words_o+1'b1;
      if(core_valid_i && core_ready_o)raw_words_o<=raw_words_o+1'b1;
      case(state)
        IDLE:if(start_i)begin instruction_address_o<=0;elapsed_cycles_o<=0;state<=FETCH;end
        FETCH:begin
          instruction<=instruction_i;index<=0;output_index<=0;skip_value<=0;
          if(instruction_i[0+:32]==0)begin state<=IDLE;done_o<=1;end
          else if(instruction_i[0+:32]==1)state<=DMA_START;
          else if(instruction_i[0+:32]<=5 && instruction_i[96+:32]!=0)state<=SCAN_READ;
          else begin invalid_o<=1;state<=IDLE;end
        end
        DMA_START:if(frame_ready && encoder_ready && !sender_pending)begin
          sent<=0;raw_index<=0;lane<=0;frame_start<=1;
          if(a4!=0)encoder_start<=1;
          state<=DMA_FEED;
        end
        DMA_FEED:begin
          if(source_valid && source_ready)sent<=sent+1'b1;
          if(sent==a2 && !sender_pending)state<=DMA_WAIT;
        end
        DMA_WAIT:if(frame_valid)begin state<=SCATTER;lane<=0;end
        SCATTER:begin
          if(lane==0)begin
            if(frame_valid)begin raw_word<=frame_data;state<=ACC_READ;end
          end else state<=ACC_READ;
        end
        ACC_READ:begin
          if(map_data_i==32'hffffffff)begin
            if(lane==0)begin lane<=1;state<=SCATTER;end
            else begin lane<=0;raw_index<=raw_index+1'b1;state<=raw_index==a6-1 ? NEXT : SCATTER;end
          end else state<=ACC_WRITE;
        end
        ACC_WRITE:begin
          if(a8!=0 && (accumulated>65'sd2147483647 || accumulated< -65'sd2147483648))invalid_o<=1;
          if(lane==0)begin lane<=1;state<=SCATTER;end
          else begin lane<=0;raw_index<=raw_index+1'b1;state<=raw_index==a6-1 ? NEXT : SCATTER;end
        end
        SCAN_READ:state<=SCAN_WAIT;
        SCAN_WAIT:begin
          value<=memory_data_i;
          state<=op==3 ? SKIP_READ : op==4 ? GAP_SEND : op==5 ? LOGIT_SEND : POST_SEND;
        end
        SKIP_READ:state<=SKIP_WAIT;
        SKIP_WAIT:begin skip_value<=memory_data_i;state<=POST_SEND;end
        POST_SEND:if(post_ready)state<=POST_WAIT;
        POST_WAIT:if(post_valid)begin store_value<=post_data;state<=STORE;end
        GAP_SEND:if(gap_ready)begin
          if(index%a4==a4-1)state<=GAP_WAIT;
          else begin index<=index+1'b1;state<=SCAN_READ;end
        end
        GAP_WAIT:if(gap_valid)begin store_value<=$signed(gap_data);state<=STORE;end
        LOGIT_SEND:if(logit_ready)state<=LOGIT_WAIT;
        LOGIT_WAIT:if(logit_valid)begin store_value<=logit_data;state<=STORE;end
        STORE:begin
          output_index<=output_index+1'b1;
          if(index==a3-1)state<=NEXT;
          else begin index<=index+1'b1;state<=SCAN_READ;end
        end
        NEXT:begin completed_instruction_o<=instruction_address_o;instruction_done_o<=1;
          instruction_address_o<=instruction_address_o+1'b1;state<=FETCH;end
        default:begin invalid_o<=1;state<=IDLE;end
      endcase
    end
  end
endmodule
