"""Build an untimed saveable harness from the maintained numeric graph bench.

Only stimulus, trace file I/O and the delayed observation check change. The
OpenEye core, graph engine, SRAM and arithmetic modules are reused unchanged.
The C++ driver clocks it and owns output streams; no OS file handle is resumed.
"""
from pathlib import Path
import re

PORTS='''module openeye_graph_resumable (
  input wire clk,rst,
  output wire input_fire_o,raw_fire_o,forward_fire_o,final_fire_o,stem_fire_o,
  output wire [63:0] input_word_o,raw_word_o,forward_word_o,
  output wire [31:0] forward_address_o,instruction_o,completed_o,last_logged_o,stem_count_o,
  output wire [7:0] stem_word_o,
  output wire [63:0] cycles_o,engine_cycles_o,elapsed_o,
  output wire setup_done_o,stem_boundary_o
);'''

SETUP='''
  // Setup ROMs are model configuration, evaluated once before inference.
  // Verilated restore retains didInit and never reruns this block.
  initial begin
    if(!$value$plusargs("PROGRAM=%s",program_path) || !$value$plusargs("PACKETS=%s",packet_path) ||
       !$value$plusargs("MAP=%s",map_path) || !$value$plusargs("COEFFICIENTS=%s",coeff_path) ||
       !$value$plusargs("INITIAL=%s",initial_path))$fatal(1,"Missing private graph ROM files");
    $readmemh(program_path,program_rom);$readmemh(packet_path,packet_rom);
    $readmemh(map_path,map_rom);$readmemh(initial_path,initial_rom);
    for(i=0;i<64;i=i+1)begin state_cycles[i]=0;state_inputs[i]=0;end
    file_coeff=$fopen(coeff_path,"r");if(!file_coeff)$fatal(1,"Missing graph coefficients");
    for(i=0;i<COEFFICIENTS;i=i+1)begin
      result=$fscanf(file_coeff,"%h %h\\n",multipliers[i],biases[i]);if(result!=2)$fatal(1,"Missing graph coefficient");
    end
    $fclose(file_coeff);
    if(USE_STEM!=0)begin
      if(INITIAL_VALUES!=3*224*224 || !$value$plusargs("STEM_WEIGHTS=%s",stem_weight_path) ||
         !$value$plusargs("STEM_COEFFICIENTS=%s",stem_coefficient_path))$fatal(1,"Missing actual stem ROMs");
      $readmemh(stem_weight_path,stem_weight_rom);file_coeff=$fopen(stem_coefficient_path,"r");
      if(!file_coeff)$fatal(1,"Missing stem coefficients");
      for(i=0;i<64;i=i+1)begin
        result=$fscanf(file_coeff,"%h %h\\n",stem_m[i],stem_b[i]);if(result!=2)$fatal(1,"Missing stem coefficient");
      end
      $fclose(file_coeff);
    end
  end
  reg [3:0] setup_phase=0;
  integer warmup=0;
  always @(negedge clk or negedge rst)begin
    if(!rst)begin
      setup_phase<=0;warmup<=0;loading<=0;load_index<=0;start<=0;
      stem_start<=0;stem_weight_write<=0;stem_coefficient_write<=0;setup_address<=0;
    end else case(setup_phase)
      0:begin
        warmup<=warmup+1;
        if(warmup==9)begin
          if(USE_STEM!=0)begin stem_weight_write<=1;setup_address<=0;setup_phase<=1;end
          else begin loading<=1;load_index<=0;setup_phase<=3;end
        end
      end
      1:begin
        if(setup_address==3*64*7*7-1)begin
          stem_weight_write<=0;stem_coefficient_write<=1;setup_address<=0;setup_phase<=2;
        end else setup_address<=setup_address+1;
      end
      2:begin
        if(setup_address==63)begin
          stem_coefficient_write<=0;loading<=1;load_index<=0;setup_phase<=3;
        end else setup_address<=setup_address+1;
      end
      3:begin
        if(load_index==INITIAL_VALUES-1)begin
          loading<=0;
          if(USE_STEM!=0)begin
            if(!stem_ready)$fatal(1,"Stem setup not ready");
            stem_start<=1;setup_phase<=4;
          end else begin start<=1;setup_phase<=5;end
        end else load_index<=load_index+1;
      end
      4:begin
        stem_start<=0;
        if(stem_outputs==200704)begin start<=1;setup_phase<=5;end
      end
      5:begin start<=0;setup_phase<=6;end
      6:begin end
      default:$fatal(1,"Unknown untimed harness setup phase");
    endcase
  end
  reg final_event_q=0;
  reg [31:0] last_logged_completed=32'hffffffff;
  assign input_fire_o=rst && accelerator.enable_dma_i_reg && core_ready;
  assign input_word_o=accelerator.data_dma_i_reg;
  assign raw_fire_o=rst && core_valid && core_sink;
  assign raw_word_o=raw;
  assign forward_fire_o=rst && write_enable && engine.instruction[0+:32]!=1;
  assign forward_address_o=memory_address;assign forward_word_o=write_data;
  assign final_fire_o=rst && final_write;assign instruction_o=instruction_address;
  assign stem_fire_o=rst && stem_valid;assign stem_word_o=stem_data;
  assign completed_o=completed;assign last_logged_o=last_logged_completed;
  assign stem_count_o=stem_outputs;assign cycles_o=64'(cycle);
  assign engine_cycles_o=engine_cycles;assign elapsed_o=elapsed;
  assign setup_done_o=setup_phase==6;
  assign stem_boundary_o=USE_STEM!=0 && stem_outputs==200704 && engine.state==1 && instruction_address==0 && engine_cycles==0;
  always @(posedge clk)if(rst)begin
    final_event_q<=final_write && memory_address==OUTPUT_BASE+OUTPUT_VALUES-1;
    if(instruction_done)last_logged_completed<=completed;
  end
  always @(negedge clk)if(rst && final_event_q)begin
    if(!observed_done || observed_invalid || observed_running || elapsed!=cycle-first_edge)
      $fatal(1,"Graph observation mismatch");
    $display("GRAPH_OUTPUT_DONE values=%0d latency=%0d slots=%0d engine_cycles=%0d",output_count,elapsed,slots,engine_cycles);$fflush();
  end
'''


def source(path=None):
    path=path or Path(__file__).with_name('rtl')/'openeye_graph_tb.sv'
    text=Path(path).read_text()
    def replace(old,new,count=1):
        nonlocal text
        if text.count(old)!=count:raise ValueError('Unknown graph bench boundary: '+old[:80])
        text=text.replace(old,new)
    replace('module openeye_graph_tb;',PORTS)
    replace('  reg clk=0,rst=1,start=0,loading=0;\n  always #5 clk=~clk;', '  reg start=0,loading=0;')
    begin=text.index('  initial begin\n');end=text.index('  always @(posedge clk)if(rst)begin\n    if(!accelerator_clock_enable',begin)
    text=text[:begin]+SETUP+'\n'+text[end:]
    replace('memory[stem_outputs]<={56\'d0,stem_data};$fwrite(stem_file,"%02h\\n",stem_data);stem_outputs<=stem_outputs+1;',
            'memory[stem_outputs]<={56\'d0,stem_data};stem_outputs<=stem_outputs+1;')
    replace('$fclose(stem_file);$display("GRAPH_STEM_DONE', '$display("GRAPH_STEM_DONE')
    replace('      if(engine.instruction[0+:32]!=1)$fwrite(file_forward,"%08h %08h %016h\\n",instruction_address,memory_address,write_data);\n','')
    replace('        $fwrite(file_output,"%016h\\n",write_data);output_count<=output_count+1;','        output_count<=output_count+1;')
    replace('actual_inputs<=actual_inputs+1;$fwrite(file_input,"%016h\\n",accelerator.data_dma_i_reg);','actual_inputs<=actual_inputs+1;')
    replace('actual_raw<=actual_raw+1;$fwrite(file_raw,"%016h\\n",raw);','actual_raw<=actual_raw+1;')
    begin=text.index('    if(final_write && memory_address==OUTPUT_BASE+OUTPUT_VALUES-1)begin\n      #1;')
    end=text.index('    if(done)begin',begin);text=text[:begin]+text[end:]
    replace('      $fclose(file_output);$fclose(file_raw);$fclose(file_input);$fclose(file_forward);\n','')
    if re.search(r'#\s*\d|\$fwrite|\$fopen\([^,]+,"[wa]"',text):raise ValueError('Timing/output OS handles remain in saveable harness')
    return text
