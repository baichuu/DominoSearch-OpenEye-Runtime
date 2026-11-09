// File-backed model ROM + synchronous scratch SRAM around the graph engine.
// Setup ROM loads happen before measurement. Inference input loading is timed.
// No expected outputs are read by this testbench or supplied to the engine.
`timescale 1ns/1ps
module openeye_graph_tb;
  parameter integer STEM_CHANNEL_LANES=1;
  parameter integer MEMORY_VALUES=16384,INITIAL_VALUES=4096,PACKET_WORDS=10000,
    MAP_LANES=1024,INSTRUCTIONS=3,COEFFICIENTS=8,OUTPUT_BASE=8192,OUTPUT_VALUES=512,
    FRAME_CAPACITY=65536,USE_STEM=0,OPTIMIZED_CORE=0;
  reg clk=0,rst=1,start=0,loading=0;
  always #5 clk=~clk;
  reg [1023:0] program_rom[0:INSTRUCTIONS-1];
  reg [63:0] packet_rom[0:PACKET_WORDS-1];
  reg [31:0] map_rom[0:MAP_LANES-1];
  reg signed [63:0] multipliers[0:COEFFICIENTS-1],biases[0:COEFFICIENTS-1];
  reg signed [63:0] initial_rom[0:INITIAL_VALUES-1],memory[0:MEMORY_VALUES-1],read_data;
  reg signed [7:0] stem_weight_rom[0:3*64*7*7-1];
  reg signed [63:0] stem_m[0:63],stem_b[0:63];
  reg stem_weight_write=0,stem_coefficient_write=0,stem_start=0;
  reg [31:0] setup_address=0;
  wire stem_ready,stem_valid,stem_done,stem_invalid;
  wire [7:0] stem_data;
  integer stem_outputs=0,stem_file=0;
  wire [31:0] instruction_address,packet_address,map_address,coefficient_address,memory_address,completed,accepted,raw_count;
  wire read_enable,write_enable,en,core_ready,core_valid,core_sink,core_last,done,invalid,instruction_done,available;
  wire [63:0] payload,raw;
  wire signed [63:0] write_data;
  wire [63:0] engine_cycles;
  wire final_write=write_enable && instruction_address==INSTRUCTIONS-2 &&
      memory_address>=OUTPUT_BASE && memory_address<OUTPUT_BASE+OUTPUT_VALUES;
  integer load_index=0,cycle=0,first_edge=-1,first_program_edge=-1,output_count=0,actual_inputs=0,actual_raw=0;
  integer i,result,file_coeff,file_output,file_raw,file_input,file_forward;
  reg [63:0] state_cycles[0:63],state_inputs[0:63];
  reg [63:0] observed_core_cycles=0,input_retry_cycles=0;
  integer state_index;
  reg [4095:0] program_path,packet_path,map_path,coeff_path,initial_path,output_path,raw_path,input_path;
  reg [4095:0] stem_weight_path,stem_coefficient_path,stem_output_path;
  reg [4095:0] forward_path;
  wire observed_running,observed_done,observed_invalid;
  wire [63:0] elapsed,slots,t0,t1,t2,t3,t4;
  // Hold the idle core clock during the scalar stem. The common observation
  // clock keeps running, so no stem cycles disappear from network latency.
  // Enable changes only at the falling edge; there are no artificial edges.
  reg accelerator_clock_enable=1;
  wire accelerator_clock=clk && accelerator_clock_enable;
  always @(negedge clk)begin
    if(USE_STEM!=0)accelerator_clock_enable<=!rst || first_edge<0 || loading || stem_outputs>=200704;
    else accelerator_clock_enable<=1;
  end
  generate if(USE_STEM!=0)begin:connected_stem
    openeye_stem_frontend #(.CHANNEL_LANES(STEM_CHANNEL_LANES)) frontend (
      .clk_i(clk),.rst_ni(rst),.image_write_i(loading),.weight_write_i(stem_weight_write),
      .write_address_i(loading ? 32'(load_index) : setup_address),
      .write_data_i(loading ? initial_rom[load_index][7:0] : stem_weight_rom[setup_address]),
      .coefficient_write_i(stem_coefficient_write),.coefficient_address_i(setup_address),
      .multiplier_i(stem_m[setup_address]),.bias_i(stem_b[setup_address]),.start_i(stem_start),
      .command_ready_o(stem_ready),.valid_o(stem_valid),.ready_i(1'b1),.data_o(stem_data),.done_o(stem_done),.invalid_o(stem_invalid));
`ifdef DOMINO_CONNECTED_STEM
    integer conv_values=0,post_values=0,pool_values=0;
    integer conv_first=-1,conv_last=-1,post_first=-1,post_last=-1,pool_first=-1,pool_last=-1;
    always @(posedge clk)if(rst)begin
      if(frontend.conv_valid && frontend.conv_ready)begin
        conv_values=conv_values+1;if(conv_first<0)conv_first=cycle;conv_last=cycle;
      end
      if(frontend.post_valid && frontend.post_ready)begin
        post_values=post_values+1;if(post_first<0)post_first=cycle;post_last=cycle;
      end
      if(stem_valid)begin
        pool_values=pool_values+1;if(pool_first<0)pool_first=cycle;pool_last=cycle;
      end
      if(start)begin
        if(conv_values!=802816 || post_values!=802816 || pool_values!=200704)$fatal(1,"Incomplete stem function coverage");
        $display("GRAPH_STEM_COMPONENT name=conv7x7 values=%0d first=%0d last=%0d",conv_values,conv_first,conv_last);
        $display("GRAPH_STEM_COMPONENT name=bn_requant_relu values=%0d first=%0d last=%0d",post_values,post_first,post_last);
        $display("GRAPH_STEM_COMPONENT name=maxpool values=%0d first=%0d last=%0d",pool_values,pool_first,pool_last);$fflush();
      end
    end
`endif
  end else begin:no_stem
    assign stem_ready=1;assign stem_valid=0;assign stem_data=0;assign stem_done=0;assign stem_invalid=0;
  end endgenerate
  openeye_graph_engine #(.FRAME_CAPACITY(FRAME_CAPACITY)) engine (
    .clk_i(clk),.rst_ni(rst),.start_i(start),.command_ready_o(available),.instruction_address_o(instruction_address),
    .instruction_i(program_rom[instruction_address]),.packet_address_o(packet_address),.packet_data_i(packet_rom[packet_address]),
    .map_address_o(map_address),.map_data_i(map_rom[map_address]),.coefficient_address_o(coefficient_address),
    .coefficient_multiplier_i(multipliers[coefficient_address]),.coefficient_bias_i(biases[coefficient_address]),
    .memory_read_o(read_enable),.memory_write_o(write_enable),.memory_address_o(memory_address),.memory_data_i(read_data),.memory_data_o(write_data),
    .enable_dma_o(en),.data_dma_o(payload),.ready_dma_i(core_ready),.core_valid_i(core_valid),.core_data_i(raw),.core_ready_o(core_sink),
    .done_o(done),.invalid_o(invalid),.instruction_done_o(instruction_done),.completed_instruction_o(completed),
    .accepted_words_o(accepted),.raw_words_o(raw_count),.elapsed_cycles_o(engine_cycles));
  // All three profiles execute the same runtime stride protocol. Some source
  // trees were frozen from stride-1 gates with a zero header default.
  OpenEye_FPGA #(.STRIDE_ENABLED(1)) accelerator (
    .clk_i(accelerator_clock),.rst_ni(rst),.enable_dma_i(en),.data_dma_i(payload),.ready_dma_o(core_ready),
    .ready_dma_i(core_sink),.enable_dma_o(core_valid),.data_dma_o(raw),.last_data_o(core_last));
`ifdef DOMINO_OPTIMIZED_CORE
  generate if(OPTIMIZED_CORE!=0)begin:optimized_guards
    always @(posedge clk)if(rst)begin
      if(accelerator.research_iact_cache_error || accelerator.research_weight_broadcast_error ||
         accelerator.psum_pipeline_inst.research_zero_bias_error)$fatal(1,"Optimized core protocol guard failed");
    end
    for(genvar cx=0;cx<2;cx=cx+1)begin:cx_guard
      for(genvar cy=0;cy<3;cy=cy+1)begin:cy_guard
        for(genvar bank=0;bank<3;bank=bank+1)begin:bank_guard
          always @(posedge clk)if(rst && accelerator.IACT_CONVERTER_X[cx].IACT_CONVERTER_Y[cy].iact_stream_constructor.BUFFER[bank].compact_address_error)
            $fatal(1,"Optimized staging mode/address guard failed");
        end
      end
    end
  end endgenerate
`endif
  openeye_network_cycle_monitor observer (
    .clk_i(clk),.rst_ni(rst),.start_valid_i(loading && load_index==0),.start_ready_i(1'b1),
    .finish_valid_i(final_write && memory_address==OUTPUT_BASE+OUTPUT_VALUES-1),.finish_ready_i(1'b1),
    .compute_active_i(!available),.postops_active_i(engine.post_valid || engine.gap_valid || engine.logit_valid),
    .transfer_active_i(loading || en || core_valid),.running_o(observed_running),.done_o(observed_done),.invalid_o(observed_invalid),
    .elapsed_cycles_o(elapsed),.total_cycles_o(slots),.compute_cycles_o(t0),.postops_cycles_o(t1),
    .transfer_cycles_o(t2),.idle_cycles_o(t3),.overlap_cycles_o(t4));
  initial begin
    if(!$value$plusargs("PROGRAM=%s",program_path) || !$value$plusargs("PACKETS=%s",packet_path) ||
       !$value$plusargs("MAP=%s",map_path) || !$value$plusargs("COEFFICIENTS=%s",coeff_path) ||
       !$value$plusargs("INITIAL=%s",initial_path) || !$value$plusargs("OUTPUT=%s",output_path) ||
       !$value$plusargs("RAW_OUTPUT=%s",raw_path) || !$value$plusargs("ACCEPTED=%s",input_path) ||
       !$value$plusargs("FORWARDED=%s",forward_path))$fatal(1,"Missing private graph files");
    $readmemh(program_path,program_rom);$readmemh(packet_path,packet_rom);$readmemh(map_path,map_rom);$readmemh(initial_path,initial_rom);
    for(i=0;i<64;i=i+1)begin state_cycles[i]=0;state_inputs[i]=0;end
    file_coeff=$fopen(coeff_path,"r");file_output=$fopen(output_path,"w");file_raw=$fopen(raw_path,"w");file_input=$fopen(input_path,"w");file_forward=$fopen(forward_path,"w");
    if(!file_coeff || !file_output || !file_raw || !file_input || !file_forward)$fatal(1,"Missing graph model/output files");
    for(i=0;i<COEFFICIENTS;i=i+1)begin
      result=$fscanf(file_coeff,"%h %h\n",multipliers[i],biases[i]);if(result!=2)$fatal(1,"Missing graph coefficient");
    end
    $fclose(file_coeff);#1;rst=0;repeat(4)@(negedge clk);rst=1;repeat(10)@(negedge clk);
    if(USE_STEM!=0)begin
      if(INITIAL_VALUES!=3*224*224 || !$value$plusargs("STEM_WEIGHTS=%s",stem_weight_path) ||
         !$value$plusargs("STEM_COEFFICIENTS=%s",stem_coefficient_path) ||
         !$value$plusargs("STEM_OUTPUT=%s",stem_output_path))$fatal(1,"Missing full stem configuration");
      $readmemh(stem_weight_path,stem_weight_rom);file_coeff=$fopen(stem_coefficient_path,"r");stem_file=$fopen(stem_output_path,"w");
      if(!file_coeff || !stem_file)$fatal(1,"Missing private stem setup files");
      for(i=0;i<64;i=i+1)begin result=$fscanf(file_coeff,"%h %h\n",stem_m[i],stem_b[i]);if(result!=2)$fatal(1,"Missing stem coefficient");end
      $fclose(file_coeff);
      for(i=0;i<3*64*7*7;i=i+1)begin stem_weight_write=1;setup_address=i;@(negedge clk);end
      stem_weight_write=0;
      for(i=0;i<64;i=i+1)begin stem_coefficient_write=1;setup_address=i;@(negedge clk);end
      stem_coefficient_write=0;
    end
    loading=1;
    for(i=0;i<INITIAL_VALUES;i=i+1)begin load_index=i;@(negedge clk);end
    loading=0;
    if(USE_STEM!=0)begin
      if(!stem_ready)$fatal(1,"Stem setup not ready");stem_start=1;@(negedge clk);stem_start=0;
      while(stem_outputs<200704)@(negedge clk);
      if(stem_invalid)$fatal(1,"Stem arithmetic/configuration failed");
    end
    start=1;@(negedge clk);start=0;
  end
  always @(posedge clk)if(rst)begin
    if(!accelerator_clock_enable && (en || core_valid || !core_ready || engine.state!=0))
      $fatal(1,"Inactive core clock may only hold an idle accelerator before graph start");
    cycle=cycle+1;if(cycle>2000000000 || invalid || stem_invalid)$fatal(1,"Graph controller timeout/invalid");
    if(loading)begin
      if(load_index==0)first_edge=cycle;
      if(USE_STEM==0)memory[load_index]<=initial_rom[load_index];
    end
    if(stem_valid)begin
      if(stem_outputs>=200704 || write_enable)$fatal(1,"Stem/graph scratch ownership overlap");
      memory[stem_outputs]<={56'd0,stem_data};$fwrite(stem_file,"%02h\n",stem_data);stem_outputs<=stem_outputs+1;
      if(stem_outputs==200703)begin $fclose(stem_file);$display("GRAPH_STEM_DONE values=200704 cycles_from_image=%0d",cycle-first_edge);$fflush();end
    end
    if(start)first_program_edge=cycle;
    if(engine.state==1)begin
      $display("GRAPH_INSTRUCTION_START index=%0d op=%0d cycles=%0d",instruction_address,program_rom[instruction_address][0+:32],engine_cycles);$fflush();
      for(state_index=0;state_index<64;state_index=state_index+1)begin state_cycles[state_index]<=0;state_inputs[state_index]<=0;end
      observed_core_cycles<=0;input_retry_cycles<=0;
    end
    if(engine.instruction[0+:32]==1 && engine.state>=2 && engine.state<=7)begin
      if(accelerator.fsm_current_state>=64)$fatal(1,"Unknown native core state counter");
      state_cycles[accelerator.fsm_current_state]<=state_cycles[accelerator.fsm_current_state]+1'b1;
      observed_core_cycles<=observed_core_cycles+1'b1;
      if(accelerator.enable_dma_i_reg && core_ready)state_inputs[accelerator.fsm_current_state]<=state_inputs[accelerator.fsm_current_state]+1'b1;
      if(accelerator.enable_dma_i_reg && !core_ready)input_retry_cycles<=input_retry_cycles+1'b1;
    end
    if((read_enable || write_enable) && memory_address>=MEMORY_VALUES)$fatal(1,"Graph scratch memory address outside allocation");
    if(read_enable)read_data<=memory[memory_address];
    if(write_enable)begin
      memory[memory_address]<=write_data;
      if(engine.instruction[0+:32]!=1)$fwrite(file_forward,"%08h %08h %016h\n",instruction_address,memory_address,write_data);
      if(final_write)begin
        if(memory_address!=OUTPUT_BASE+output_count)$fatal(1,"Final tensor order/coverage mismatch");
        $fwrite(file_output,"%016h\n",write_data);output_count<=output_count+1;
      end
    end
    if(accelerator.enable_dma_i_reg && core_ready)begin
      actual_inputs<=actual_inputs+1;$fwrite(file_input,"%016h\n",accelerator.data_dma_i_reg);
    end
    if(core_valid && core_sink)begin actual_raw<=actual_raw+1;$fwrite(file_raw,"%016h\n",raw);end
    if(instruction_done)begin
      $display("GRAPH_INSTRUCTION_DONE index=%0d cycles=%0d inputs=%0d raw_words=%0d",completed,engine_cycles,accepted,raw_count);$fflush();
      if(engine.instruction[0+:32]==1)begin
        for(state_index=0;state_index<64;state_index=state_index+1)
          if(state_cycles[state_index]!=0 || state_inputs[state_index]!=0)
            $display("GRAPH_CORE_STATE index=%0d state=%0d cycles=%0d input_words=%0d",completed,state_index,state_cycles[state_index],state_inputs[state_index]);
        $display("GRAPH_CORE_INTERVAL index=%0d cycles=%0d retry_cycles=%0d",completed,observed_core_cycles,input_retry_cycles);$fflush();
      end
    end
    if(final_write && memory_address==OUTPUT_BASE+OUTPUT_VALUES-1)begin
      #1;if(!observed_done || observed_invalid || observed_running || elapsed!=cycle-first_edge)$fatal(1,"Graph observation mismatch");
      $display("GRAPH_OUTPUT_DONE values=%0d latency=%0d slots=%0d engine_cycles=%0d",output_count,elapsed,slots,engine_cycles);$fflush();
    end
    if(done)begin
      if(output_count!=OUTPUT_VALUES || actual_inputs!=accepted || actual_raw!=raw_count || !observed_done || observed_invalid || !core_ready)$fatal(1,"Incomplete actual graph execution");
      $fclose(file_output);$fclose(file_raw);$fclose(file_input);$fclose(file_forward);
      $display("GRAPH_PROGRAM_DONE instructions=%0d inputs=%0d raw_words=%0d cycles=%0d",instruction_address,accepted,raw_count,engine_cycles);$finish;
    end
  end
endmodule
