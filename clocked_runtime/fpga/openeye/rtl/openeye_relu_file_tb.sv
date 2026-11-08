// Simulation bridge to the pinned OpenEye activation block. The input source
// and output sink are always available. No DMA/board latency is implied.
`timescale 1ns/1ps
module openeye_relu_file_tb;
  parameter WIDTH=64;
  parameter LANES=4;
  parameter DEPTH=1;
  reg clk=0;
  always #5 clk=~clk;
  reg rst_n=0;
  reg relu_en=1;
  reg [WIDTH*LANES-1:0] data_i=0;
  wire [WIDTH*LANES-1:0] data_o;
  af_cluster #(.WIDTH(WIDTH),.DEPTH(DEPTH),.VECTOR_SIZE(LANES)) dut (
    .clk_i(clk),.rst_ni(rst_n),.relu_en_i(relu_en),.data_i(data_i),.data_o(data_o));
  string input_path, output_path;
  integer input_file, output_file, count, i, lane, groups, cycle, status, bubbles;
  reg [WIDTH-1:0] word;
  initial begin
    if (!$value$plusargs("INPUT=%s",input_path) || !$value$plusargs("OUTPUT=%s",output_path) ||
        !$value$plusargs("COUNT=%d",count) || !$value$plusargs("RELU=%d",relu_en) ||
        !$value$plusargs("BUBBLES=%d",bubbles)) $fatal(1,"Missing bridge argument");
    if (count<=0 || WIDTH<2 || LANES<=0 || DEPTH!=1 || bubbles<0) $fatal(1,"Unsupported bridge geometry");
    input_file=$fopen(input_path,"r");output_file=$fopen(output_path,"w");
    if (!input_file || !output_file) $fatal(1,"Cannot open private tensor files");
    repeat(2) @(negedge clk);
    rst_n=1;
    groups=(count+LANES-1)/LANES;
    cycle=0;
    for(i=0;i<groups;i=i+1) begin
      // Explicit idle cycles allow accounting controls without inventing DMA.
      repeat(bubbles) begin @(posedge clk);#1;cycle=cycle+1;@(negedge clk);end
      data_i=0;
      for(lane=0;lane<LANES;lane=lane+1) begin
        if(i*LANES+lane<count) begin
          status=$fscanf(input_file,"%h",word);
          if(status!=1) $fatal(1,"Missing input word");
          data_i[lane*WIDTH+:WIDTH]=word;
        end
      end
      @(posedge clk);#1;cycle=cycle+1;
      for(lane=0;lane<LANES;lane=lane+1)
        if(i*LANES+lane<count) $fdisplay(output_file,"%h",data_o[lane*WIDTH+:WIDTH]);
      @(negedge clk);
    end
    status=$fscanf(input_file,"%h",word);
    if(status==1) $fatal(1,"Extra input word");
    $fclose(input_file);$fclose(output_file);
    $display("DOMINO_RELU_RTL_DONE count=%0d groups=%0d cycles=%0d depth=%0d bubbles=%0d",count,groups,cycle,DEPTH,bubbles);
    $finish;
  end
endmodule
