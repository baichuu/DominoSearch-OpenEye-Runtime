// File driver for real post-op -> optional GAP RTL with a continuous clock.
// Only operands enter the DUT; golden outputs are checked outside the simulator.
`timescale 1ns/1ps
module openeye_postops_elastic_tb;
  parameter integer FIFO_DEPTH=8;
  parameter integer GAP_PIXELS=0;
  parameter integer CHANNEL_GROUPS=1;
  parameter integer MAX_VALUES=8192;
  reg clk=0, rst=1;
  always #5 clk=~clk;
  reg valid=0, sink_ready=0;
  wire input_ready, post_valid, post_ready, output_valid;
  wire signed [63:0] post_data;
  wire [63:0] output_data;
  wire post_input_valid,post_input_ready;
  wire signed [63:0] accumulated_value;
  wire [31:0] accumulator_index;
  wire accumulator_done,accumulator_invalid;
  reg command_start=0;
  reg signed [63:0] values[0:MAX_VALUES-1], skips[0:MAX_VALUES-1];
  reg signed [63:0] multipliers[0:MAX_VALUES-1], biases[0:MAX_VALUES-1], skip_multipliers[0:MAX_VALUES-1];
  reg [5:0] shifts[0:MAX_VALUES-1];
  reg residuals[0:MAX_VALUES-1], relus[0:MAX_VALUES-1];
  integer count, sink_pattern, source_pattern;
  integer incoming=0, outgoing=0, cycle=0, first_edge=-1;
  integer input_stalls=0, output_stalls=0;
  integer in_file, out_file, result, i;
  reg blocked_input=0, blocked_output=0;
  reg [63:0] held_output;
  reg [4095:0] input_path, output_path;
  wire running, done, invalid;
  wire [63:0] elapsed, slots, compute_ticks, post_ticks, transfer_ticks, idle_ticks, overlap_ticks;
  wire [31:0] post_rom_index=CHANNEL_GROUPS>1 ? count-count/CHANNEL_GROUPS+accumulator_index : incoming;

  generate if (CHANNEL_GROUPS>1) begin : WITH_ACCUMULATION
    wire signed [31:0] accumulator_data;
    wire command_ready;
    openeye_psum_accumulator_stream #(.CAPACITY(MAX_VALUES)) accumulator (
      .clk_i(clk),.rst_ni(rst),.start_i(command_start),.command_ready_o(command_ready),
      .elements_i(32'(count/CHANNEL_GROUPS)),.groups_i(32'(CHANNEL_GROUPS)),
      .valid_i(valid),.ready_o(input_ready),.data_i(values[incoming][31:0]),
      .valid_o(post_input_valid),.ready_i(post_input_ready),.data_o(accumulator_data),
      .index_o(accumulator_index),.done_o(accumulator_done),.invalid_o(accumulator_invalid));
    assign accumulated_value={{32{accumulator_data[31]}},accumulator_data};
  end else begin : WITHOUT_ACCUMULATION
    assign input_ready=post_input_ready;
    assign post_input_valid=valid;
    assign accumulated_value=values[incoming];
    assign accumulator_index=0;
    assign accumulator_done=0;
    assign accumulator_invalid=0;
  end endgenerate

  openeye_postops_elastic #(.FIFO_DEPTH(FIFO_DEPTH)) post (
    .clk_i(clk),.rst_ni(rst),.valid_i(post_input_valid),.ready_o(post_input_ready),
    .value_i(accumulated_value),.skip_i(skips[post_rom_index]),
    .multiplier_i(multipliers[post_rom_index]),.bias_i(biases[post_rom_index]),
    .skip_multiplier_i(skip_multipliers[post_rom_index]),.shift_i(shifts[post_rom_index]),
    .residual_i(residuals[post_rom_index]),.relu_i(relus[post_rom_index]),
    .valid_o(post_valid),.ready_i(post_ready),.data_o(post_data));
  generate if (GAP_PIXELS>0) begin : WITH_GAP
    openeye_gap_elastic gap (
      .clk_i(clk),.rst_ni(rst),.valid_i(post_valid),.ready_o(post_ready),
      .data_i(post_data[7:0]),.pixels_i(16'(GAP_PIXELS)),
      .valid_o(output_valid),.ready_i(sink_ready),.data_o(output_data));
  end else begin : WITHOUT_GAP
    assign post_ready=sink_ready;
    assign output_valid=post_valid;
    assign output_data=post_data;
  end endgenerate
  localparam integer GROUP_SIZE=GAP_PIXELS>0 ? GAP_PIXELS : 1;
  wire first_input=valid && incoming==0;
  wire final_output=output_valid && outgoing==count/(GROUP_SIZE*CHANNEL_GROUPS)-1;
  openeye_network_cycle_monitor observer (
    .clk_i(clk),.rst_ni(rst),.start_valid_i(first_input),.start_ready_i(input_ready),
    .finish_valid_i(final_output),.finish_ready_i(sink_ready),
    .compute_active_i(CHANNEL_GROUPS>1 && valid && input_ready),.postops_active_i(post_input_valid && post_input_ready || post_valid),
    .transfer_active_i(output_valid),.running_o(running),.done_o(done),.invalid_o(invalid),
    .elapsed_cycles_o(elapsed),.total_cycles_o(slots),.compute_cycles_o(compute_ticks),
    .postops_cycles_o(post_ticks),.transfer_cycles_o(transfer_ticks),
    .idle_cycles_o(idle_ticks),.overlap_cycles_o(overlap_ticks));

  initial begin
    if (!$value$plusargs("INPUT=%s",input_path) || !$value$plusargs("OUTPUT=%s",output_path) ||
        !$value$plusargs("COUNT=%d",count) || !$value$plusargs("SINK_PATTERN=%d",sink_pattern) ||
        !$value$plusargs("SOURCE_PATTERN=%d",source_pattern)) $fatal(1,"Missing driver config");
    if (count<1 || count>MAX_VALUES || count%(GROUP_SIZE*CHANNEL_GROUPS)!=0) $fatal(1,"Invalid driver geometry");
    in_file=$fopen(input_path,"r");out_file=$fopen(output_path,"w");
    if (!in_file || !out_file) $fatal(1,"Cannot open private files");
    for (i=0;i<count;i=i+1) begin
      result=$fscanf(in_file,"%h %h %h %h %h %d %d %d\n",values[i],skips[i],multipliers[i],biases[i],skip_multipliers[i],shifts[i],residuals[i],relus[i]);
      if (result!=8) $fatal(1,"Missing actual operands");
    end
    $fclose(in_file);
    #1;rst=0;#1;rst=1;
    @(negedge clk);command_start=1;
    @(negedge clk);command_start=0;
  end
  always @(negedge clk) if (rst) begin
    valid=incoming<count && (blocked_input || source_pattern==0 || cycle%7!=0);
    // Long burst + periodic stalls exercise credit exhaustion and FIFO wrap.
    sink_ready=sink_pattern==0 || (cycle%97>49 && cycle%5!=0);
  end
  always @(posedge clk) if (rst) begin
    cycle=cycle+1;
    if (cycle>100000) $fatal(1,"No forward progress");
    blocked_input<=valid && !input_ready;
    if (valid && !input_ready) input_stalls=input_stalls+1;
    if (valid && input_ready) begin
      if (incoming==0) first_edge=cycle;
      incoming<=incoming+1;
    end
    if (blocked_output && (!output_valid || output_data!==held_output))
      $fatal(1,"Output changed while backpressured");
    blocked_output<=output_valid && !sink_ready;
    held_output<=output_data;
    if (output_valid && !sink_ready) output_stalls=output_stalls+1;
    if (output_valid && sink_ready) begin
      $fwrite(out_file,"%016h\n",output_data);
      outgoing<=outgoing+1;
      if (outgoing==count/(GROUP_SIZE*CHANNEL_GROUPS)-1) begin
        #1;
        if (!done || invalid || accumulator_invalid || running || incoming!=count || elapsed!=cycle-first_edge)
          $fatal(1,"Invalid continuous component timeline");
        $fclose(out_file);
        $display("ELASTIC_DONE inputs=%0d outputs=%0d latency=%0d slots=%0d input_stalls=%0d output_stalls=%0d",incoming,outgoing,elapsed,slots,input_stalls,output_stalls);
        $finish;
      end
    end
  end
endmodule
