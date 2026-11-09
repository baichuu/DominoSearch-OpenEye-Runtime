"""Validate measured per-function latencies and an exclusive full-network ledger.

Function rows overlap: only exclusive rows may be summed to network latency.
No tensor or coefficient enters the public tables.
"""
import re
from fpga.openeye.graph_function_metrics import FUNCTIONS,FSM_NAMES
from fpga.openeye.validate_graph_program import require

FUNCTION_COLUMNS=('scope','site','instruction','function','status','latency_unit','transactions','latency_min_cycles',
    'latency_max_cycles','latency_sum_cycles','tensor_span_cycles','busy_slots',
    'input_stall_cycles','output_stall_cycles','additive_to_network')
STEP_COLUMNS=('scope','site','instruction','step','cycles','additive_to_network')
METRIC_KEYS={'scope','instruction','function','inputs','outputs','first','last','span',
    'latency_sum','latency_min','latency_max','busy_slots','input_stalls','output_stalls'}


def fields(line,prefix,keys):
    tokens=line[len(prefix):].split();require(tokens and all(re.fullmatch('[a-z_]+=[0-9]+',v)for v in tokens),'Malformed measured metric')
    parts=[v.split('=')for v in tokens];require(len(parts)==len({k for k,_ in parts}),'Duplicate measured metric field')
    value={k:int(v)for k,v in parts};require(set(value)==keys,'Missing/unknown measured metric fields');return value


def parse(log,program,prep,*,stop_after=None,network_cycles=None):
    completed=len(program)-1 if stop_after is None else stop_after+1
    require(0<=completed<len(program),'Invalid metrics prefix boundary')
    stages=prep.get('stages',[]);sites={}
    for stage in stages:
        for index in range(stage['first_instruction'],stage['last_instruction']+1):sites[index]=stage['layer']
    for i,row in enumerate(program[:completed]):
        if row[0]==3:
            previous=[s for s in stages if s['last_instruction']<i]
            name=previous[-1]['layer']if previous else'instruction_'+str(i)
            sites[i]=name.split('.downsample.')[0].split('.conv')[0]+'.residual'
        elif row[0]==4:sites[i]='gap'
        sites.setdefault(i,'instruction_'+str(i))
    expected={(0,0xffffffff,12):1}
    if prep.get('connected_stem'):
        expected.update({(0,0xffffffff,k):v for k,v in zip(range(1,8),(802816,0,802816,802816,0,802816,802816))})
        expected.update({(0,0xffffffff,10):1,(0,0xffffffff,11):1})
    for i,row in enumerate(program[:completed]):
        if row[0]in(2,3):
            n=row[3];residual=row[0]==3;relu=residual or bool(row[6])
            sizes=(0 if residual else n,n if residual else 0,n,n if relu else 0,0 if relu else n,n,n)
            expected.update({(1,i,k):v for k,v in enumerate(sizes,1)})
        elif row[0]==4:expected[1,i,8]=row[3]//row[4]
        elif row[0]==5:expected[1,i,9]=row[3]
    metrics={};ledger={};totals=[];stems=[]
    for line in log.splitlines():
        if line.startswith('GRAPH_FUNCTION_METRIC '):
            v=fields(line,'GRAPH_FUNCTION_METRIC ',METRIC_KEYS);key=v['scope'],v['instruction'],v['function']
            require(key in expected and key not in metrics,'Foreign/duplicate/uncompleted function record')
            count=expected[key];require(v['inputs']==v['outputs']==count,'Incomplete function transaction coverage')
            if count:
                require(v['last']>=v['first']and v['span']==v['last']-v['first']and
                    v['latency_min']<=v['latency_max']<=v['span']and
                    count*v['latency_min']<=v['latency_sum']<=count*v['latency_max']and
                    0<v['busy_slots']<=v['span']+1,'Inconsistent function timing counters')
                if key[2]in(1,2,3,4,5,6,9):
                    require(v['latency_min']==v['latency_max']==1 and v['latency_sum']==count,'Wrong measured register boundary latency')
                elif key[2]==7:require(v['latency_min']>=1,'FIFO timing cannot bypass the real output register')
                elif key[2]==12:require(v['span']==prep['initial_values']-1,'Image ingress coverage differs from actual input length')
            else:require(all(v[k]==0 for k in('first','last','span','latency_min','latency_max','latency_sum','busy_slots')),'Unused function has fabricated work')
            metrics[key]=v
        elif line.startswith('GRAPH_STEP_LEDGER '):
            v=fields(line,'GRAPH_STEP_LEDGER ',{'instruction','state','cycles'});key=v['instruction'],v['state']
            require(key not in ledger and 0<=key[0]<completed and 1<=key[1]<20 and v['cycles']>0,'Invalid exclusive controller ledger')
            op=program[key[0]][0]
            allowed={1:{1,2,3,4,5,6,7,19},2:{1,8,9,12,13,14,19},3:{1,8,9,10,11,12,13,14,19},
                     4:{1,8,9,15,16,14,19},5:{1,8,9,17,18,14,19}}
            require(key[1]in allowed[op],'Controller state contradicts executed function')
            ledger[key]=v['cycles']
        elif line.startswith('GRAPH_FUNCTION_TOTAL '):
            totals.append(fields(line,'GRAPH_FUNCTION_TOTAL ',{'image','stem_compute','stem_fill_drain','launch','graph','total'}))
        elif line.startswith('GRAPH_STEM_LEDGER '):
            stems.append(fields(line,'GRAPH_STEM_LEDGER ',{'image','compute','fill_drain'}))
    require(set(metrics)==set(expected),'Missing per-function measurements')
    require({i for i,_ in ledger}==set(range(completed)),'Missing instruction ledger coverage')
    starts=[tuple(map(int,v))for v in re.findall(r'^GRAPH_INSTRUCTION_START index=(\d+) op=(\d+) cycles=(\d+)$',log,re.M)]
    ends=[tuple(map(int,v))for v in re.findall(r'^GRAPH_INSTRUCTION_DONE index=(\d+) cycles=(\d+) inputs=\d+ raw_words=\d+$',log,re.M)]
    require([i for i,_,_ in starts[:completed]]==list(range(completed))and
            [i for i,_ in ends]==list(range(completed)),'Missing/reordered measured instruction intervals')
    for i,end in ends:
        require(starts[i][1]==program[i][0],'Measured instruction differs from actual program')
        # The final NEXT edge follows final logit storage and is outside the
        # network interval. Earlier instruction NEXT edges are included.
        expected_cycles=end-starts[i][2]-int(stop_after is None and i==completed-1)
        require(sum(c for(ip,_),c in ledger.items()if ip==i)==expected_cycles,
                'Exclusive instruction states do not match actual instruction interval')
    require(len(stems)==1 and stems[0]['image']==prep['initial_values']-1,'Missing/invalid image and stem exclusive ledger')
    functions=[]
    for key,v in sorted(metrics.items()):
        scope,ip,fn=key;site=('stem'if scope==0 else sites[ip])
        functions.append(dict(scope='stem'if scope==0 else'graph',site=site,instruction=-1 if scope==0 else ip,
            function=FUNCTIONS[fn],status='MEASURED'if v['outputs']else'NOT_EXECUTED',
            latency_unit='per_value'if fn in(1,2,3,4,5,6,7,9)else'per_channel_reduction'if fn==8 else'whole_tensor',
            transactions=v['outputs'],latency_min_cycles=v['latency_min'],
            latency_max_cycles=v['latency_max'],latency_sum_cycles=v['latency_sum'],tensor_span_cycles=v['span'],
            busy_slots=v['busy_slots'],input_stall_cycles=v['input_stalls'],output_stall_cycles=v['output_stalls'],additive_to_network=False))
    grouped={}
    for(ip,state),c in sorted(ledger.items()):
        key=sites[ip],FSM_NAMES[state];grouped[key]=grouped.get(key,0)+c
    steps=[dict(scope='graph',site=site,instruction=-1,step=step,cycles=c,additive_to_network=True)
           for(site,step),c in grouped.items()]
    for step,key in [('image_ingress','image'),('stem_compute_with_overlap','compute'),('stem_fill_drain','fill_drain')]:
        steps.append(dict(scope='stem',site='stem',instruction=-1,step=step,cycles=stems[0][key],additive_to_network=True))
    certified_sum=None
    if stop_after is None:
        require(len(totals)==1 and network_cycles is not None,'Missing final actual-cycle closure')
        t=totals[0]
        require(t['image']==stems[0]['image']and t['stem_compute']==stems[0]['compute']and t['stem_fill_drain']==stems[0]['fill_drain'],
                'Changed completed stem ledger')
        require(sum(ledger.values())==t['graph']and sum(v for k,v in t.items()if k!='total')==t['total']==network_cycles,
                'Exclusive steps do not equal continuous actual network cycles')
        steps.append(dict(scope='graph',site='dispatch',instruction=-1,step='graph_launch',cycles=t['launch'],additive_to_network=True))
        certified_sum=sum(r['cycles']for r in steps);require(certified_sum==network_cycles,'Exported exclusive rows do not close')
    else:require(not totals,'Final total cannot be assigned to a prefix')
    return dict(functions=functions,exclusive_steps=steps,exclusive_cycles_sum=certified_sum,instruction_ledger_rows=len(ledger),
                measurement_scope='passive actual register handshakes; function rows overlap; sum only exclusive steps')
