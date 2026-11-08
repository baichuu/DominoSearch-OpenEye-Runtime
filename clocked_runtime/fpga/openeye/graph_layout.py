"""Geometry/address contract shared by the clocked experiment's audit tools."""
import numpy as np
from fpga.openeye.experiments.halo_stripe_dispatch import stripe_plan


def encode_instruction(op,*args):
    if len(args)>31:raise ValueError('Too many controller instruction fields')
    return ''.join(f'{int(v)&0xffffffff:08x}'for v in reversed([op,*args]+[0]*(31-len(args))))+'\n'


def windows(width,height,stride):
    logical=[(width+stride-1)//stride,(height+stride-1)//stride]
    if [width,height,stride]==[56,56,2]:
        return[dict(kind='halo',**w,source_input=[width,height],source_stride=stride,
            execution_input=[w['execution_width'],height],execution_stride=stride,logical_output=logical)
            for w in stripe_plan('stripe24')]
    # The native 28x28 stride-2 constructor has a verified raw mismatch in
    # persistent execution. Far-pad to32; the original14x14 output remains
    # exact SAME correlation and all extra raw positions stay checked.
    multiple=8 if (width,height,stride)==(28,28,2)else 4 if stride==2 else 2
    return[dict(kind='native',source_input=[width,height],source_stride=stride,
        execution_input=[(v+multiple-1)//multiple*multiple for v in(width,height)],execution_stride=stride,
        logical_output=logical,output_start=0,output_stop=logical[0])]


def map_input(activation,window):
    a=np.asarray(activation,dtype=np.int64)
    if list(a.shape[1:])!=window['source_input']:raise ValueError('Wrong live source geometry')
    if window['kind']=='halo':a=a[:,window['input_start']:window['input_stop'],:]
    elif window['kind']!='native':raise ValueError('Unsupported clocked spatial lowering')
    target=window['execution_input']
    if any(x>y for x,y in zip(a.shape[1:],target)):raise ValueError('Invalid halo/far padding')
    return np.pad(a,((0,0),(0,target[0]-a.shape[1]),(0,target[1]-a.shape[2])))
