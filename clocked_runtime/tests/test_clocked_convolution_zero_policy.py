"""Fail-closed generated binding controls; native numeric proof is separate."""
from pathlib import Path
import pytest
from fpga.openeye.clocked_core_compat import retain_zeros_for_clocked_convolution


def existing_wiring(root):
    hdl=root/'hdl';hdl.mkdir()
    for name in('OpenEye_Parallel.v','OpenEye_Cluster.v','PE_cluster.v','PE.v','data_pipeline_iact.v'):
        (hdl/name).write_text('research_retain_zero_values_i\n')
    top=hdl/'OpenEye_FPGA.v'
    top.write_text('.research_retain_zero_values_i(!fully_connected_layer && kernel_size_x == 3 && stride_x == 2 && iact_size_y == 32),\n')
    return top


def test_all_convolution_geometries_retain_values_but_leave_fc_and_metadata(tmp_path):
    top=existing_wiring(tmp_path);before={p.name:p.read_bytes()for p in top.parent.iterdir()if p!=top}
    retain_zeros_for_clocked_convolution(tmp_path);retain_zeros_for_clocked_convolution(tmp_path)
    assert top.read_text()=='.research_retain_zero_values_i(!fully_connected_layer && kernel_size_x == 3),\n'
    assert before=={p.name:p.read_bytes()for p in top.parent.iterdir()if p!=top}


@pytest.mark.parametrize('mutation',['unknown','duplicate','missing-port'])
def test_unknown_or_partially_applied_retention_policy_is_rejected(tmp_path,mutation):
    top=existing_wiring(tmp_path)
    if mutation=='unknown':top.write_text('.research_retain_zero_values_i(unknown),\n')
    elif mutation=='duplicate':top.write_text(top.read_text()*2)
    else:(top.parent/'PE.v').write_text('unknown')
    with pytest.raises(ValueError):retain_zeros_for_clocked_convolution(tmp_path)
