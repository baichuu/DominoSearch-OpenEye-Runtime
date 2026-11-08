"""Retain zero values while preserving sparse address/tag SPAD encoding.

Pin9a's SPARSITY_EN=0 also clears the address SPAD. Using that compile-time
mode as a zero-retention switch corrupts a sparse PE. This separate parameter
only changes value acceptance; all sparse address metadata remains enabled.
Generated copies only. Native/raw FC needs its own mixed-protocol gate.
"""
from pathlib import Path


def apply(root):
    root=Path(root);path=root/'hdl/data_pipeline_iact.v';source=path.read_text()
    needle='parameter SPARSITY_EN               = 1,'
    if source.count(needle)!=1 or 'RETAIN_ZERO_VALUES' in source:raise ValueError('Unknown activation pipeline')
    source=source.replace(needle,'parameter RETAIN_ZERO_VALUES        = 0,\n    '+needle)
    if source.count('(!SPARSITY_EN)')!=4 or source.count('if (!SPARSITY_EN) begin')!=1:
        raise ValueError('Unknown zero-value guards')
    source=source.replace('(!SPARSITY_EN)','(!SPARSITY_EN || RETAIN_ZERO_VALUES)',3)
    if source.count('if (!SPARSITY_EN) begin')!=1:raise ValueError('Address SPAD mode must stay unchanged')
    path.write_text(source)
    path=root/'hdl/PE.v';source=path.read_text()
    needle='data_pipeline_iact #(\n'
    if source.count(needle)!=1:raise ValueError('Unknown PE activation pipeline binding')
    source=source.replace(needle,needle+'      .RETAIN_ZERO_VALUES(1),\n')
    path.write_text(source)
