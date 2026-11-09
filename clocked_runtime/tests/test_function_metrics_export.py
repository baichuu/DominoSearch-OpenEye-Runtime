"""Synthetic publication controls; these fixtures are not measured RTL evidence."""
import csv
import json
import pytest
from fpga.openeye.audit_function_metrics import FUNCTION_COLUMNS,STEP_COLUMNS
from fpga.openeye.seal_clocked_public_evidence import seal


def evidence(root):
    network=root/'network';network.mkdir(parents=True)
    function=dict(zip(FUNCTION_COLUMNS,('graph','layer1.0.conv1',1,'relu','MEASURED','per_value',2,1,1,2,10,4,0,0,False)))
    step=dict(zip(STEP_COLUMNS,('graph','layer1.0.conv1',-1,'post_wait',23,True)))
    report=dict(result='PASS_INDEPENDENT_COMPLETE_CLOCKED_NETWORK_AUDIT',full_network_rtl_pass=True,
        full_network_cycles=23,function_metrics_protocol='passive-function-cycles-v1',functions=[function],
        exclusive_steps=[step],exclusive_cycles_sum=23,measured_instruction_ledger_rows=1,
        function_measurement_scope='synthetic test fixture',
        stages=[dict(layer='layer1.0.conv1',scheme='4:4',commands=1,checked_values=2,latency_cycles=23,
            convolution_dma_accumulation_cycles=13,postops_cycles=10,output_sha256='a'*64)],
        postops=[dict(operation='bn_relu',site='layer1.0.conv1',checked_values=2,latency_cycles=10)],
        native_core_states=[dict(instruction=0,state=1,cycles=13,input_words=1)])
    (network/'audit.json').write_text(json.dumps(report))
    (network/'junit.xml').write_text('<testsuite tests="1" failures="0"><testcase/></testsuite>')
    for name,key,columns in [('layers.csv','stages',None),('postops.csv','postops',None),
            ('native-core-states.csv','native_core_states',None),('functions.csv','functions',FUNCTION_COLUMNS),
            ('exclusive-steps.csv','exclusive_steps',STEP_COLUMNS)]:
        with (network/name).open('w',newline='')as stream:
            writer=csv.DictWriter(stream,fieldnames=columns or list(report[key][0]));writer.writeheader();writer.writerows(report[key])
    return report


def test_measured_tables_preserve_counters_without_publishing_private_files(tmp_path):
    root=tmp_path/'private';report=evidence(root);(root/'private-operands.bin').write_bytes(b'private')
    output=tmp_path/'public';manifest=seal(root,output,'optimized','a'*40,'123')
    assert manifest['full_network_rtl_pass']
    assert json.loads((output/'network/audit.json').read_text())==report
    assert (output/'network/functions.csv').is_file()and (output/'network/exclusive-steps.csv').is_file()
    assert not (output/'private-operands.bin').exists()


@pytest.mark.parametrize('mutation',['different-csv','missing-csv','overlap','wrong-sum','partial','negative-cycles'])
def test_inconsistent_function_tables_are_not_published(tmp_path,mutation):
    root=tmp_path/'private';report=evidence(root)
    if mutation=='different-csv':
        path=root/'network/functions.csv';path.write_text(path.read_text().replace(',2,1,1,2,',',2,1,1,99,'))
    elif mutation=='missing-csv':(root/'network/exclusive-steps.csv').unlink()
    elif mutation=='overlap':report['functions'][0]['additive_to_network']=True
    elif mutation=='wrong-sum':report['exclusive_cycles_sum']=24
    elif mutation=='partial':del report['functions']
    else:
        report['exclusive_steps'][0]['cycles']=-1;report['exclusive_cycles_sum']=-1;report['full_network_cycles']=-1
    (root/'network/audit.json').write_text(json.dumps(report))
    with pytest.raises(ValueError):seal(root,tmp_path/'public','optimized','a'*40,'123')
    assert not (tmp_path/'public').exists()
