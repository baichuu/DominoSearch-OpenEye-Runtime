"""Fabricated counter rejection controls; never claim DUT/network evidence."""
import pytest
from fpga.openeye.audit_function_metrics import parse


def fixture():
    lines=[]
    def metric(scope,ip,fn,count,first=5,span=10,latency=1):
        values=dict(scope=scope,instruction=ip,function=fn,inputs=count,outputs=count,
            first=first if count else 0,last=first+span if count else 0,span=span if count else 0,
            latency_sum=count*latency,latency_min=latency if count else 0,latency_max=latency if count else 0,
            busy_slots=4 if count else 0,input_stalls=0,output_stalls=0)
        lines.append('GRAPH_FUNCTION_METRIC '+' '.join(f'{k}={v}'for k,v in values.items()))
    metric(0,0xffffffff,12,1,0,3,3)
    for fn in range(1,8):metric(1,0,fn,0 if fn in(2,5)else 2)
    for state,count in[(1,1),(8,2),(9,2),(12,2),(13,10),(14,2)]:
        lines.append(f'GRAPH_STEP_LEDGER instruction=0 state={state} cycles={count}')
    lines+=['GRAPH_STEM_LEDGER image=3 compute=0 fill_drain=0',
        'GRAPH_INSTRUCTION_START index=0 op=2 cycles=0',
        'GRAPH_INSTRUCTION_DONE index=0 cycles=20 inputs=0 raw_words=0',
        'GRAPH_FUNCTION_TOTAL image=3 stem_compute=0 stem_fill_drain=0 launch=1 graph=19 total=23']
    return '\n'.join(lines),[[2,0,10,2,0,2,1,24],[0]],dict(initial_values=4)


def test_only_exclusive_rows_close_and_function_rows_are_nonadditive():
    log,program,prep=fixture();report=parse(log,program,prep,network_cycles=23)
    assert report['exclusive_cycles_sum']==23
    assert sum(v['cycles']for v in report['exclusive_steps'])==23
    assert all(not row['additive_to_network']for row in report['functions'])
    assert all(row['additive_to_network']for row in report['exclusive_steps'])
    assert next(v for v in report['functions']if v['function']=='activation_bypass')['status']=='NOT_EXECUTED'


@pytest.mark.parametrize('mutation',['missing-function','duplicate-function','wrong-count','wrong-latency',
    'wrong-span','unknown-function','bad-state','wrong-total','missing-ledger','duplicate-field','extra-field'])
def test_missing_or_inconsistent_function_evidence_is_rejected(mutation):
    log,program,prep=fixture()
    if mutation=='missing-function':log='\n'.join(v for v in log.splitlines()if 'function=3 'not in v)
    elif mutation=='duplicate-function':log+='\n'+next(v for v in log.splitlines()if 'function=3 'in v)
    elif mutation=='wrong-count':log=log.replace('inputs=2 outputs=2','inputs=1 outputs=1',1)
    elif mutation=='wrong-latency':log=log.replace('latency_sum=2 latency_min=1 latency_max=1','latency_sum=4 latency_min=2 latency_max=2',1)
    elif mutation=='wrong-span':log=log.replace('span=10','span=9',1)
    elif mutation=='unknown-function':log=log.replace('function=3 ','function=100 ')
    elif mutation=='bad-state':log=log.replace('state=8 cycles=2','state=7 cycles=2')
    elif mutation=='wrong-total':log=log.replace('total=23','total=24')
    elif mutation=='missing-ledger':log='\n'.join(v for v in log.splitlines()if 'state=13 'not in v)
    elif mutation=='duplicate-field':log=log.replace('function=3 ','function=3 function=3 ')
    else:log=log.replace('function=3 ','function=3 payload=999 ')
    with pytest.raises(ValueError):parse(log,program,prep,network_cycles=23)


def test_prefix_cannot_inherit_a_final_cycle_total():
    log,program,prep=fixture()
    with pytest.raises(ValueError):parse(log,program,prep,stop_after=0)
    prefix='\n'.join(v for v in log.splitlines()if not v.startswith('GRAPH_FUNCTION_TOTAL '))
    prefix+='\nGRAPH_STEP_LEDGER instruction=0 state=19 cycles=1'
    report=parse(prefix,program,prep,stop_after=0)
    assert report['exclusive_cycles_sum']is None


def test_counter_redistribution_cannot_hide_behind_correct_global_total():
    log,program,prep=fixture()
    log=log.replace('index=0 cycles=20','index=0 cycles=21')
    with pytest.raises(ValueError,match='instruction interval'):parse(log,program,prep,network_cycles=23)
