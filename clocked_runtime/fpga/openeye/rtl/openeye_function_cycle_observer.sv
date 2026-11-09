// Passive, ordered transaction timing. No data, clock gating or ready outputs.
// Clock-edge distances and occupied observation slots are separate quantities.
`timescale 1ns/1ps
module openeye_function_cycle_observer #(
  parameter integer SCOPE=0,FUNCTION_ID=0,DEPTH=16,EXPECTED_LATENCY=-1
) (
  input wire clk_i,rst_ni,clear_i,dump_i,
  input wire [31:0] instruction_i,
  input wire input_valid_i,input_ready_i,output_valid_i,output_ready_i
);
  reg [63:0] clock_edge,inputs,outputs,first_edge,last_edge,latency_sum,latency_min,latency_max;
  reg [63:0] busy_slots,input_stalls,output_stalls;
  reg [63:0] timestamps[0:DEPTH-1];
  integer read_ptr,write_ptr,pending;
  wire input_fire=input_valid_i && input_ready_i;
  wire output_fire=output_valid_i && output_ready_i;
  wire [63:0] measured_latency=pending==0 ? 64'd0 : clock_edge-timestamps[read_ptr];
  always @(posedge clk_i or negedge rst_ni)begin
    if(!rst_ni)begin
      clock_edge<=0;inputs<=0;outputs<=0;first_edge<=0;last_edge<=0;
      latency_sum<=0;latency_min<=0;latency_max<=0;busy_slots<=0;input_stalls<=0;output_stalls<=0;
      read_ptr<=0;write_ptr<=0;pending<=0;
    end else begin
      clock_edge<=clock_edge+1;
      // Dump reads completed old statistics when clear_i starts the next scope.
      if(dump_i)begin
        if(pending!=0 || inputs!=outputs)$fatal(1,"Incomplete passive function timing");
        $display("GRAPH_FUNCTION_METRIC scope=%0d instruction=%0d function=%0d inputs=%0d outputs=%0d first=%0d last=%0d span=%0d latency_sum=%0d latency_min=%0d latency_max=%0d busy_slots=%0d input_stalls=%0d output_stalls=%0d",
          SCOPE,instruction_i,FUNCTION_ID,inputs,outputs,first_edge,last_edge,
          outputs==0 ? 64'd0 : last_edge-first_edge,latency_sum,latency_min,latency_max,busy_slots,input_stalls,output_stalls);
      end
      if(clear_i)begin
        if(pending!=0 || input_fire || output_fire)$fatal(1,"Function scope reset while active");
        inputs<=0;outputs<=0;first_edge<=0;last_edge<=0;latency_sum<=0;latency_min<=0;latency_max<=0;
        busy_slots<=0;input_stalls<=0;output_stalls<=0;read_ptr<=0;write_ptr<=0;pending<=0;
      end else begin
        if(pending!=0 || input_fire)busy_slots<=busy_slots+1;
        if(input_valid_i && !input_ready_i)input_stalls<=input_stalls+1;
        if(output_valid_i && !output_ready_i)output_stalls<=output_stalls+1;
        if(input_fire)begin
          if(pending==DEPTH && !output_fire)$fatal(1,"Function timestamp capacity exceeded");
          timestamps[write_ptr]<=clock_edge;write_ptr<=write_ptr==DEPTH-1 ? 0 : write_ptr+1;
          inputs<=inputs+1;if(inputs==0)first_edge<=clock_edge;
        end
        if(output_fire)begin
          if(pending==0 && !input_fire)$fatal(1,"Function completion without accepted input");
          if(EXPECTED_LATENCY>=0 && measured_latency!=EXPECTED_LATENCY)$fatal(1,"Unexpected actual function register latency");
          if(pending!=0 && clock_edge<timestamps[read_ptr])$fatal(1,"Function clock moved backwards");
          read_ptr<=read_ptr==DEPTH-1 ? 0 : read_ptr+1;
          outputs<=outputs+1;last_edge<=clock_edge;latency_sum<=latency_sum+measured_latency;
          if(outputs==0 || measured_latency<latency_min)latency_min<=measured_latency;
          if(outputs==0 || measured_latency>latency_max)latency_max<=measured_latency;
        end
        case({input_fire,output_fire})
          2'b10:pending<=pending+1;
          2'b01:pending<=pending-1;
          default:begin end
        endcase
      end
    end
  end
endmodule
