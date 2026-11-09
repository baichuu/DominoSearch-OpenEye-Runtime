"""Passive graph instrumentation; no input data, arithmetic or handshakes change."""

FUNCTIONS={1:'bn_affine',2:'residual_scale_add',3:'requant_round',4:'relu',
           5:'activation_bypass',6:'int8_saturation',7:'postop_output_fifo',
           8:'gap_reduce_divide_round',9:'fc_scale_bias_q40',10:'stem_convolution',
           11:'maxpool3',12:'image_ingress'}
FSM_NAMES=['idle','fetch','dma_start','dma_feed','dma_wait','scatter','acc_read',
           'acc_write','scan_read','scan_wait','skip_read','skip_wait','post_send',
           'post_wait','store','gap_send','gap_wait','logit_send','logit_wait','next']


def transaction(name,function,scope,start,finish,dump,clear="1'b0",instruction="32'hffffffff"):
    return f'''
openeye_function_cycle_observer #(.FUNCTION_ID({function}),.SCOPE({scope})) {name}(
 .clk_i(clk),.rst_ni(rst),.clear_i({clear}),.dump_i({dump}),.instruction_i({instruction}),
 .input_valid_i({start}),.input_ready_i(1'b1),.output_valid_i({finish}),.output_ready_i(1'b1));
'''


def postops(name,path,scope,dump,clear="1'b0",instruction="32'hffffffff"):
    return f'''
openeye_postops_cycle_metrics #(.SCOPE({scope})) {name}(
 .clk_i(clk),.rst_ni(rst),.clear_i({clear}),.dump_i({dump}),.instruction_i({instruction}),
 .valid_i({path}.valid_i),.ready_i({path}.ready_o),.residual_i({path}.residual_i),
 .valid_s1_i({path}.core.valid_s1),.valid_s2_i({path}.core.valid_s2),.valid_s3_i({path}.core.valid_s3),
 .relu_s2_i({path}.core.relu_s2),.core_valid_i({path}.core.valid_o),
 .output_valid_i({path}.valid_o),.output_ready_i({path}.ready_i));
'''


def inject(source,*,connected_stem=False,stem_lanes=1):
    if source.count('endmodule')!=1 or 'begin:function_metrics'in source:
        raise ValueError('Unknown graph instrumentation boundary')
    stem='connected_stem.frontend'
    text='''
generate if(FUNCTION_METRICS!=0)begin:function_metrics
  wire stem_compute_active;
  reg measuring=0,total_reported=0;
  reg [63:0] image_ticks=0,stem_mac_ticks=0,stem_tail_ticks=0,launch_ticks=0,graph_ticks=0;
  reg [63:0] states[0:19];
  integer metric_state;
'''
    text+=transaction('image',12,0,'loading && load_index==0','loading && load_index==INITIAL_VALUES-1','start')
    if connected_stem:
        branch='scalar'if stem_lanes==1 else'channel_parallel'
        text+=f'assign stem_compute_active={stem}.{branch}.conv.active && !{stem}.{branch}.conv.draining;\n'
        text+='begin:stem\n'
        text+=postops('postops',stem+'.post',0,'start')
        text+=transaction('convolution',10,0,stem+'.start_fire',
            f'{stem}.conv_valid && {stem}.conv_ready && {stem}.conv_filter==63 && {stem}.conv_x==111 && {stem}.conv_y==111','start')
        text+=transaction('pool',11,0,
            f'{stem}.post_valid && {stem}.post_ready && {stem}.pool.channel==0 && {stem}.pool.row==1 && {stem}.pool.col==1',
            'stem_valid && stem_outputs==200703','start')
        text+='end\n'
    else:text+='assign stem_compute_active=0;\n'
    text+=postops('postops','engine.post',1,'instruction_done && (engine.instruction[0+:32]==2 || engine.instruction[0+:32]==3)',
                 'engine.state==1','completed')
    text+=transaction('gap',8,1,'engine.gap.valid_i && engine.gap.ready_o && engine.gap.index==0',
        'engine.gap.valid_o && engine.gap.ready_i','instruction_done && engine.instruction[0+:32]==4','engine.state==1','completed')
    text+=transaction('logit',9,1,'engine.logit.valid_i && engine.logit.ready_o',
        'engine.logit.valid_o && engine.logit.ready_i','instruction_done && engine.instruction[0+:32]==5','engine.state==1','completed')
    text+='''
  // One exclusive bucket per edge AFTER first image acceptance through final
  // logit storage. These alone may be summed to the network edge-distance.
  always @(posedge clk or negedge rst)begin
    if(!rst)begin
      measuring<=0;total_reported<=0;image_ticks<=0;stem_mac_ticks<=0;stem_tail_ticks<=0;launch_ticks<=0;graph_ticks<=0;
      for(metric_state=0;metric_state<20;metric_state=metric_state+1)states[metric_state]<=0;
    end else begin
      if(loading && load_index==0)measuring<=1;
      if(measuring)begin
        if(loading)image_ticks<=image_ticks+1;
        else if(USE_STEM!=0 && stem_outputs<200704)begin
          if(stem_compute_active)stem_mac_ticks<=stem_mac_ticks+1;
          else stem_tail_ticks<=stem_tail_ticks+1;
        end else if(engine.state==0)launch_ticks<=launch_ticks+1;
        else if(engine.state>=1 && engine.state<=19)graph_ticks<=graph_ticks+1;
        else $fatal(1,"Unclassified measured graph edge");
      end
      if(final_write && memory_address==OUTPUT_BASE+OUTPUT_VALUES-1)measuring<=0;
      if(engine.state==1)begin
        for(metric_state=0;metric_state<20;metric_state=metric_state+1)states[metric_state]<=0;
        states[1]<=measuring ? 1 : 0;
      end else if(measuring && engine.state>1 && engine.state<=19)states[engine.state]<=states[engine.state]+1;
      if(instruction_done)begin
        for(metric_state=0;metric_state<20;metric_state=metric_state+1)
          if(states[metric_state]!=0)
            $display("GRAPH_STEP_LEDGER instruction=%0d state=%0d cycles=%0d",completed,metric_state,states[metric_state]);
      end
      if(start)begin
        $display("GRAPH_STEM_LEDGER image=%0d compute=%0d fill_drain=%0d",image_ticks,stem_mac_ticks,stem_tail_ticks);
      end
      if(observed_done && !total_reported)begin
        if(measuring || image_ticks+stem_mac_ticks+stem_tail_ticks+launch_ticks+graph_ticks!=elapsed)
          $fatal(1,"Exclusive function ledger does not sum to actual network latency");
        $display("GRAPH_FUNCTION_TOTAL image=%0d stem_compute=%0d stem_fill_drain=%0d launch=%0d graph=%0d total=%0d",
          image_ticks,stem_mac_ticks,stem_tail_ticks,launch_ticks,graph_ticks,elapsed);
        total_reported<=1;
      end
    end
  end
end endgenerate
'''
    return source.rsplit('endmodule',1)[0]+text+'endmodule\n'
