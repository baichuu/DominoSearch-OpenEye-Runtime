"""Fabricated ingress/timeline rejection controls, never model RTL evidence."""
import json
from pathlib import Path
import pytest
import numpy as np

from fpga.openeye.graph_layout import encode_instruction
from fpga.openeye.validate_graph_program import validate,sha
from fpga.openeye.audit_clocked_graph import parse_timeline
from fpga.openeye.graph_layout import windows,map_input
from fpga.openeye.runtime_activations import convolution


def fixture(root):
    root.mkdir()
    rows=encode_instruction(1,0,2,0,0,0,2,64,0,0,1,1,64,0,1,1,0,1,1,0,0,4)+encode_instruction(2,64,68,4,0,4,1,24)+encode_instruction(0)
    (root/'program.hex').write_text(rows)
    (root/'maps.hex').write_text('00000000\n00000001\n00000002\n00000003\n')
    (root/'packets.hex').write_text('0000000000000000\n'*2)
    (root/'initial-memory.hex').write_text('0000000000000000\n'*64)
    (root/'coefficients.hex').write_text('0000000000000001 0000000000000000\n')
    prep=dict(instructions=3,packet_words=2,map_lanes=4,coefficient_count=1,initial_values=64,
        memory_values=72,output_base=68,output_values=4,raw_words=2,
        files={p.name:sha(p)for p in root.iterdir()})
    (root/'preparation.json').write_text(json.dumps(prep));return prep


def test_structural_fixture_does_not_certify_any_dut(tmp_path):
    fixture(tmp_path/'gate');report=validate(tmp_path/'gate',whole=False)
    assert report['result']=='VALID_STRUCTURAL_GRAPH_NOT_DUT_PASS'
    assert report['full_network_cycles']is None and report['full_network_rtl_pass']is False
    with pytest.raises((ValueError,KeyError)):validate(tmp_path/'gate')


@pytest.mark.parametrize('kernel',[1,3])
def test_far32_keeps_all_original_stride2_values_and_signed_weights(kernel):
    rng=np.random.default_rng(48);a=rng.integers(0,128,(64,28,28),dtype=np.int64)
    w=rng.integers(-127,128,(64,8,kernel,kernel),dtype=np.int64)
    window=windows(28,28,2)[0];assert window['execution_input']==[32,32]
    mapped=map_input(a,window);reference=convolution(a,w,2)
    if kernel==1:
        embedded=np.zeros((64,8,3,3),dtype=np.int64);embedded[:,:,1,1]=w[:,:,0,0];w=embedded
    assert np.array_equal(convolution(mapped,w,2)[:,:14,:14],reference)


@pytest.mark.parametrize('mode',['changed-packet','duplicate-map','missing-map','bad-op','out-of-memory','bad-groups','zero-gap'])
def test_reject_incomplete_or_inconsistent_controller_inputs(tmp_path,mode):
    root=tmp_path/'gate';prep=fixture(root)
    if mode=='changed-packet':(root/'packets.hex').write_text('0000000000000001\n'*2)
    elif mode=='duplicate-map':(root/'maps.hex').write_text('00000000\n'*4)
    elif mode=='missing-map':(root/'maps.hex').write_text('00000000\n')
    else:
        lines=(root/'program.hex').read_text().splitlines()
        words=[(int(lines[0],16)>>(i*32))&0xffffffff for i in range(32)]
        if mode=='bad-op':words[0]=6
        if mode=='out-of-memory':words[7]=1000
        if mode=='bad-groups':words[12]=3
        if mode=='zero-gap':words=[4,0,64,4,0]+[0]*27
        lines[0]=encode_instruction(words[0],*words[1:]).strip();(root/'program.hex').write_text('\n'.join(lines)+'\n')
    if mode!='changed-packet':
        prep['files']={name:sha(root/name)for name in prep['files']};(root/'preparation.json').write_text(json.dumps(prep))
    with pytest.raises(ValueError):validate(root,whole=False)


def timeline():
    return '''GRAPH_INSTRUCTION_START index=0 op=1 cycles=0
GRAPH_INSTRUCTION_START index=1 op=2 cycles=10
GRAPH_INSTRUCTION_DONE index=0 cycles=10 inputs=2 raw_words=2
GRAPH_CORE_STATE index=0 state=1 cycles=3 input_words=2
GRAPH_CORE_STATE index=0 state=10 cycles=5 input_words=0
GRAPH_CORE_INTERVAL index=0 cycles=8 retry_cycles=0
GRAPH_INSTRUCTION_START index=2 op=0 cycles=20
GRAPH_INSTRUCTION_DONE index=1 cycles=20 inputs=2 raw_words=2
'''


def test_fabricated_timeline_requires_complete_state_and_word_counts():
    parse_timeline(timeline(),[[1,0,2,0,0,0,2],[2],[0]])


@pytest.mark.parametrize('mode',['missing-completion','missing-states','wrong-words','wrong-state-sum','wrong-raw','overlap','error'])
def test_reject_partial_timeline_or_native_counter_failure(mode):
    text=timeline()
    if mode=='missing-completion':text='\n'.join(v for v in text.splitlines()if 'DONE index=1'not in v)
    elif mode=='missing-states':text='\n'.join(v for v in text.splitlines()if 'CORE_INTERVAL'not in v)
    elif mode=='wrong-words':text=text.replace('state=1 cycles=3 input_words=2','state=1 cycles=3 input_words=1')
    elif mode=='wrong-state-sum':text=text.replace('state=10 cycles=5','state=10 cycles=4')
    elif mode=='wrong-raw':text=text.replace('cycles=10 inputs=2 raw_words=2','cycles=10 inputs=2 raw_words=1')
    elif mode=='overlap':text=text.replace('op=2 cycles=10','op=2 cycles=9')
    else:text+='\n%Error: actual RTL failed\n'
    with pytest.raises(ValueError):parse_timeline(text,[[1,0,2,0,0,0,2],[2],[0]])
