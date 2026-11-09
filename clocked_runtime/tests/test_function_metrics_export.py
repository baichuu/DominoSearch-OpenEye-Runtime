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
        model='mac20',optimized=True,stem_channel_lanes=3,binary_sha256='b'*64,preparation_sha256='c'*64,
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
    table=output/'network/function-tables/full-network-by-site.csv'
    with table.open()as stream:rows=list(csv.DictReader(stream))
    assert sum(int(row['total_cycles'])for row in rows)==23
    assert 'network/function-tables/tables.md'in manifest['files']
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


def test_readable_sites_keep_controller_dma_and_sram_in_total(tmp_path):
    from fpga.openeye.summarize_function_metrics import summarize
    report=evidence(tmp_path/'private')
    parts=[('stem','image_ingress',3),('stem','stem_compute_with_overlap',10),
        ('stem','stem_fill_drain',2),('layer1.0.conv1','dma_feed',5),
        ('layer1.0.conv1','acc_write',1),('layer1.0.conv1','scan_wait',1),
        ('layer1.0.conv1','post_wait',1),('layer1.0.conv1','store',1),
        ('layer1.0.residual','post_send',2),('gap','gap_wait',2),
        ('fc','logit_wait',1),('dispatch','graph_launch',1)]
    report['exclusive_steps']=[dict(scope='stem'if site=='stem'else'graph',site=site,instruction=-1,
        step=step,cycles=cycles,additive_to_network=True)for site,step,cycles in parts]
    report['full_network_cycles']=report['exclusive_cycles_sum']=30
    rows=summarize(report);assert sum(row['total_cycles']for row in rows)==30
    conv=next(row for row in rows if row['site']=='layer1.0.conv1')
    assert conv['total_cycles']==9 and conv['dma_feed_wait']==5 and conv['sram_store']==1
    assert rows[0]['site']=='stem'and rows[-1]['site']=='dispatch'
    report['exclusive_steps'][0]['step']='unknown_unmeasured_work'
    with pytest.raises(ValueError,match='Unclassified'):summarize(report)


def test_pipelined_report_explains_overlapping_sram_writes(tmp_path):
    root=tmp_path/'private';report=evidence(root);report['postops_pipelined']=True
    (root/'network/audit.json').write_text(json.dumps(report))
    output=tmp_path/'public';seal(root,output,'optimized','a'*40,'123')
    text=(output/'network/function-tables/tables.md').read_text()
    assert 'giá trị 0 không có nghĩa là không ghi dữ liệu'in text
    assert json.loads((output/'network/audit.json').read_text())['postops_pipelined']is True
