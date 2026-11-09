// Continuous actual RTL frontend test. Model setup precedes the measured
// interval; every image byte load, compute/pad cycle and output stall is timed.
`timescale 1ns/1ps
module openeye_stem_frontend_tb;
  parameter integer WIDTH=224,HEIGHT=224,CHANNELS=3,FILTERS=64,KERNEL=7,STRIDE=2,SHIFT=24;
  parameter integer CHANNEL_LANES=1;
  localparam integer OUTPUTS=FILTERS*(((WIDTH+STRIDE-1)/STRIDE+1)/2)*(((HEIGHT+STRIDE-1)/STRIDE+1)/2);
  reg clk=0,rst=1;
  always #5 clk=~clk;
  reg image_write=0,weight_write=0,coefficient_write=0,start=0,sink_ready=1;
  reg [31:0] address=0,coefficient_address=0;
  reg signed [7:0] byte_data=0;
  reg signed [63:0] multiplier=0,bias=0;
  wire command_ready,valid,done,invalid;
  wire [7:0] data;
  integer weights_file,image_file,coefficient_file,output_file,i,result;
  integer convolution_file=0,conv_outputs=0;
  reg [4095:0] convolution_path;
  integer cycle=0,first_edge=-1,outputs=0,pattern,output_stalls=0;
  reg blocked_output=0;
  reg [7:0] held_data;
  reg [4095:0] weights_path,image_path,coefficients_path,output_path;
  wire observing,observed_done,observed_invalid;
  wire [63:0] elapsed,slots,t0,t1,t2,t3,t4;
  openeye_stem_frontend #(.WIDTH(WIDTH),.HEIGHT(HEIGHT),.CHANNELS(CHANNELS),.FILTERS(FILTERS),
    .KERNEL(KERNEL),.STRIDE(STRIDE),.SHIFT(SHIFT),.CHANNEL_LANES(CHANNEL_LANES)) dut (
    .clk_i(clk),.rst_ni(rst),.image_write_i(image_write),.weight_write_i(weight_write),
    .write_address_i(address),.write_data_i(byte_data),.coefficient_write_i(coefficient_write),
    .coefficient_address_i(coefficient_address),.multiplier_i(multiplier),.bias_i(bias),
    .start_i(start),.command_ready_o(command_ready),.valid_o(valid),.ready_i(sink_ready),
    .data_o(data),.done_o(done),.invalid_o(invalid));
  openeye_network_cycle_monitor observer (
    .clk_i(clk),.rst_ni(rst),.start_valid_i(image_write && address==0),.start_ready_i(1'b1),
    .finish_valid_i(valid && outputs==OUTPUTS-1),.finish_ready_i(sink_ready),
    .compute_active_i(!command_ready),.postops_active_i(dut.post_valid),.transfer_active_i(image_write || valid),
    .running_o(observing),.done_o(observed_done),.invalid_o(observed_invalid),
    .elapsed_cycles_o(elapsed),.total_cycles_o(slots),.compute_cycles_o(t0),.postops_cycles_o(t1),
    .transfer_cycles_o(t2),.idle_cycles_o(t3),.overlap_cycles_o(t4));
  initial begin
    if(!$value$plusargs("WEIGHTS=%s",weights_path) || !$value$plusargs("IMAGE=%s",image_path) ||
       !$value$plusargs("COEFFICIENTS=%s",coefficients_path) || !$value$plusargs("OUTPUT=%s",output_path) ||
       !$value$plusargs("SINK_PATTERN=%d",pattern))$fatal(1,"Missing private paths");
    weights_file=$fopen(weights_path,"r");image_file=$fopen(image_path,"r");
    coefficient_file=$fopen(coefficients_path,"r");output_file=$fopen(output_path,"w");
    if($value$plusargs("CONVOLUTION_OUTPUT=%s",convolution_path))begin
      convolution_file=$fopen(convolution_path,"w");
      if(!convolution_file)$fatal(1,"Missing private raw convolution output");
    end
    if(!weights_file || !image_file || !coefficient_file || !output_file)$fatal(1,"Missing private data");
    #1;rst=0;#1;rst=1;
    for(i=0;i<CHANNELS*FILTERS*KERNEL*KERNEL;i=i+1)begin
      @(negedge clk);weight_write=1;address=i;result=$fscanf(weights_file,"%h\n",byte_data);
      if(result!=1)$fatal(1,"Missing weight");
    end
    @(negedge clk);weight_write=0;
    for(i=0;i<FILTERS;i=i+1)begin
      @(negedge clk);coefficient_write=1;coefficient_address=i;
      result=$fscanf(coefficient_file,"%h %h\n",multiplier,bias);
      if(result!=2)$fatal(1,"Missing coefficient");
    end
    @(negedge clk);coefficient_write=0;
    for(i=0;i<CHANNELS*WIDTH*HEIGHT;i=i+1)begin
      @(negedge clk);image_write=1;address=i;result=$fscanf(image_file,"%h\n",byte_data);
      if(result!=1)$fatal(1,"Missing image byte");
    end
    @(negedge clk);image_write=0;start=1;
    @(negedge clk);start=0;
    $fclose(weights_file);$fclose(image_file);$fclose(coefficient_file);
  end
  always @(negedge clk)if(rst)sink_ready=pattern==0 || (cycle%97>49 && cycle%5!=0);
  always @(posedge clk)if(rst)begin
    cycle=cycle+1;
    if(image_write && address==0)first_edge=cycle;
    if(cycle>300000000)$fatal(1,"Frontend timeout");
    if(invalid)$fatal(1,"Invalid frontend command/arithmetic");
    if(dut.conv_valid && dut.conv_ready)begin
      if(convolution_file)$fwrite(convolution_file,"%08h\n",dut.conv_data);
      conv_outputs<=conv_outputs+1;
    end
    if(blocked_output && (!valid || data!==held_data))$fatal(1,"Stalled output changed");
    blocked_output<=valid && !sink_ready;held_data<=data;
    if(valid && !sink_ready)output_stalls=output_stalls+1;
    if(valid && sink_ready)begin
      $fwrite(output_file,"%02h\n",data);outputs<=outputs+1;
      if(outputs==OUTPUTS-1)begin
        #1;
        if(!observed_done || observed_invalid || observing || elapsed!=cycle-first_edge)
          $fatal(1,"Frontend timeline differs from actual edges");
        $fclose(output_file);
        if(convolution_file)$fclose(convolution_file);
        if(conv_outputs!=FILTERS*((WIDTH+STRIDE-1)/STRIDE)*((HEIGHT+STRIDE-1)/STRIDE))
          $fatal(1,"Incomplete convolution output before activation");
        $display("STEM_FRONTEND_DONE outputs=%0d latency=%0d slots=%0d stalls=%0d",outputs,elapsed,slots,output_stalls);
        $finish;
      end
    end
  end
endmodule
