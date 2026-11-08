"""Generated-copy compatibility for persistent mixed-geometry commands."""
from pathlib import Path
import re


def gate_unused_pooling_ring(root):
    path=Path(root)/'hdl/OpenEye_FPGA.v';source=path.read_text()
    old='.ready_i(pooling_buffer_enable),'
    new='.ready_i(pooling_buffer_enable && pooling),'
    if source.count(new)==1 and old not in source:return
    if source.count(old)!=1 or new in source:raise ValueError('Unknown native pooling ring connection')
    # GET_QUANTIZE pulses this enable even for convolution/FC. When geometry
    # shrinks across commands, drifting ring pointers can collide. The ring
    # participates only in native pooling commands; this graph uses its own
    # real RTL stem pool. Never suppress SRAM assertions or change RAM modes.
    path.write_text(source.replace(old,new))


def retain_zeros_for_bounded_stride2(root):
    """Keep sparse tags but retain values on the failed32-square stride2 path.

    Other shapes retain the optimized sparse-anchor behavior. No weight or
    arithmetic change; no SRAM assertion or output comparison is disabled.
    """
    root=Path(root);files={name:(root/'hdl'/name).read_text()for name in
        ('OpenEye_FPGA.v','OpenEye_Parallel.v','OpenEye_Cluster.v','PE_cluster.v','PE.v','data_pipeline_iact.v')}
    marker='research_retain_zero_values_i'
    if all(marker in text for text in files.values()):return
    if any(marker in text for text in files.values()):raise ValueError('Partially applied scoped zero-retention policy')
    def replace(name,old,new):
        require=files[name].count(old)==1
        if not require:raise ValueError('Unknown scoped zero-retention source: '+name)
        files[name]=files[name].replace(old,new)
    for name in('OpenEye_Parallel.v','OpenEye_Cluster.v','PE_cluster.v','PE.v','data_pipeline_iact.v'):
        pattern=r'^[ \t]*input[ \t]+(?:wire[ \t]+)?clk_i[ \t]*,[ \t]*$'
        if len(re.findall(pattern,files[name],re.M))!=1:raise ValueError('Unknown scoped-policy module interface')
        files[name]=re.sub(pattern,lambda m:m[0]+'\n    input wire research_retain_zero_values_i,',files[name],flags=re.M)
    replace('OpenEye_FPGA.v','.research_accumulate_i(research_accumulate),',
        '.research_accumulate_i(research_accumulate),\n        .research_retain_zero_values_i(!fully_connected_layer && kernel_size_x == 3 && stride_x == 2 && iact_size_y == 32),')
    replace('OpenEye_Parallel.v','.research_accumulate_i(research_accumulate_i),',
        '.research_accumulate_i(research_accumulate_i),\n          .research_retain_zero_values_i(research_retain_zero_values_i),')
    replace('OpenEye_Cluster.v','.accumulate_research_i(research_accumulate_i),',
        '.accumulate_research_i(research_accumulate_i),\n      .research_retain_zero_values_i(research_retain_zero_values_i),')
    replace('PE_cluster.v','.accumulate_research_i(accumulate_research_i),',
        '.accumulate_research_i(accumulate_research_i),\n            .research_retain_zero_values_i(research_retain_zero_values_i),')
    replace('PE.v',') iact_data_handler (',') iact_data_handler (\n      .research_retain_zero_values_i(research_retain_zero_values_i),')
    name='data_pipeline_iact.v';text=files[name]
    if text.count('(!SPARSITY_EN)')!=4 or text.count('if (!SPARSITY_EN) begin')!=1:raise ValueError('Unknown sparse-address retention format')
    files[name]=text.replace('(!SPARSITY_EN)','(!SPARSITY_EN || research_retain_zero_values_i)',3)
    if 'if (!SPARSITY_EN) begin'not in files[name]:raise ValueError('Sparse address metadata must remain enabled')
    for name,text in files.items():(root/'hdl'/name).write_text(text)
