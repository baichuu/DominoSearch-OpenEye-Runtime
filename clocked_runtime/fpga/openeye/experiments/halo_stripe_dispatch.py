"""Geometry-only spatial stripes with exact stride2 halo/crop coordinates."""
import json
import os
from pathlib import Path

APPEND='''
from domino_halo_stripe_dispatch import configured_sequence as _halo_sequence
_before_halo_sequence = run_sequence
async def run_sequence(dut,callback):
    return await _halo_sequence(dut,callback,_before_halo_sequence)
'''


def stripe_plan(layout):
    if layout not in ('stripe24','stripe12'):
        raise ValueError('Only explicit original56/output28 stripe layouts are supported')
    width=24 if layout=='stripe24'else 12
    result=[]
    for start in range(0,28,width):
        stop=min(start+width,28)
        origin=0 if start==0 else 2*start-2
        end=min(56,2*stop)
        size=(end-origin+3)//4*4
        crop=start-origin//2
        result.append(dict(index=len(result),output_start=start,output_stop=stop,
            input_start=origin,input_stop=end,execution_width=size,execution_height=56,
            crop_start=crop,crop_stop=crop+stop-start,far_zero_padding=size-(end-origin)))
    return result


async def configured_sequence(dut,callback,original):
    async def configured(handle):
        pattern=json.loads(Path(os.environ['OPENEYE_WGHT_PATTERN_JSON']).read_text())
        layout=pattern['research_halo_layout'];window=pattern['research_halo_window']
        expected=stripe_plan(layout)
        if type(window.get('index'))is not int or not 0<=window['index']<len(expected) or window!=expected[window['index']]:
            raise AssertionError('Halo/crop plan differs from the declared source geometry')
        e=pattern['execution']
        if not (e['kernel_width']==e['kernel_height']==3 and e['stride']==2
                and e['input_channels']==64 and e['output_channels']==16
                and e['input_width']==window['execution_width']and e['input_height']==56):
            raise AssertionError('Unsupported halo runtime parameters')
        settings=dict(INPUT_SIZE_X=e['input_width'],INPUT_SIZE_Y=e['input_height'],
            INPUT_CHANNELS=64,NUM_FILTERS=16,KERNEL_SIZE_X=3,KERNEL_SIZE_Y=3,STRIDE=2)
        os.environ.update({k:str(v)for k,v in settings.items()})
        print('DOMINO_HALO_CONFIG_PASS '+json.dumps(dict(window=window,settings=settings),sort_keys=True),flush=True)
        await callback(handle)
    await original(dut,configured)
