"""Offline native OpenEye command mapper for the clocked graph controller.

Each instance binds one frozen generated source tree. Run different source
trees in separate processes: upstream modules cache their import directory.
"""
import ast
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import numpy as np


class GraphMapper:
    def __init__(self, source, optimized=False):
        self.source=Path(source).resolve();self.optimized=optimized;self.cache_key=None
        sys.path[:0]=[str(self.source/'src'),str(self.source/'test/cocotb_fpga'),str(self.source/'.domino-inputs')]
        header=self.source/'sim/parameters_FPGA.vh'
        for k,v in re.findall(r'parameter\s+(\w+)\s*=\s*(\d+)',header.read_text()):os.environ[k]=v
        os.environ.update(LOGGER_LEVEL='50',BUFFER_WIDTH_IACT_STREAM_CONSTRUCTOR='16',RAM_CELLS='32',DATAFLOW='row_stationary',
                          OPENEYE_ARCHITECTURE='optimized-network-v2' if optimized else 'full-layer-r3')
        from open_eye.open_eye_parameters import OpenEyeParameters
        from open_eye.layer_parameters import LayerParameters
        from open_eye.conv_mapper import ConvMapper
        from open_eye.dense_mapper import DenseMapper
        from open_eye.test_utils_main import calculate_conv_serial
        self.hardware=OpenEyeParameters(serial=True);self.Parameter=LayerParameters
        self.Conv=ConvMapper;self.Dense=DenseMapper;self.coordinates=calculate_conv_serial
        self.layouts={}
        # Install the EXACT maintained fixture encoder embedded in the frozen
        # upstream compatibility bench. Do not fall back to the upstream zero
        # skipper: it loses trailing regions and invalidates prefix transport.
        bench=self.source/'test/cocotb_fpga/OpenEye_FPGA_tb.py'
        tree=ast.parse(bench.read_text())
        controls=[n for n in ast.walk(tree)if isinstance(n,ast.If)and
                  '_domino_pair_safe' in ast.unparse(n.test)and isinstance(n.test,ast.UnaryOp)]
        if len(controls)!=1:raise ValueError('Missing/ambiguous maintained pair-safe encoder')
        from open_eye.wght_stream_mapper import ConvWghtStreamMapper
        from fpga.openeye.nm_offset_adapter import decode_region,encode_sparse_positions_paired
        import logging
        namespace=dict(ConvWghtStreamMapper=ConvWghtStreamMapper,decode_region=decode_region,
                       encode_sparse_positions_paired=encode_sparse_positions_paired,os=os,logger=logging.getLogger('graph-packing'))
        code=compile(ast.Module(body=[controls[0]],type_ignores=[]),str(bench),'exec')
        exec(code,namespace)
        self.encoder_source_sha256=hashlib.sha256(bench.read_bytes()).hexdigest()

    def parameters(self,width,height,filters,kernel,stride,linear=False):
        name='dense'if linear else'conv2d'
        layer=SimpleNamespace(name=name,filters=filters,
            input=SimpleNamespace(shape=(1,64)if linear else(1,width,height,64)),
            output=SimpleNamespace(shape=(1,filters)if linear else(1,(width+stride-1)//stride,(height+stride-1)//stride,filters)),
            kernel=SimpleNamespace(shape=(64,filters)if linear else(kernel,kernel,64,filters)),
            kernel_size=(kernel,kernel),strides=(stride,stride))
        topology={'Previous_Layer':{name:['input_layer']},'Next_Layer':{name:[]}}
        p=self.Parameter({},layer,self.hardware,topology,0)
        p.choose_iact_storage_input=0;p.choose_iact_storage_output=0
        if p.needed_total_transmissions!=1:raise ValueError('Graph command must fit one native transmission')
        return p

    def command(self,activation,weights,pattern,cache_key,linear=False):
        a=np.asarray(activation,dtype=np.int64);w=np.asarray(weights,dtype=np.int64)
        if linear:
            width,height,filters,kernel,stride=64,1,w.shape[0],1,1
            content=[[a.reshape(-1).tolist()],w.tolist(),[0]*filters]
        else:
            e=pattern['execution'];width,height=a.shape[1:];filters=w.shape[1];kernel=w.shape[2];stride=e['stride']
            content=[[a.tolist()],w.tolist(),[0]*filters]
        p=self.parameters(width,height,filters,kernel,stride,linear)
        mapper_class=self.Dense if linear else self.Conv
        def write_stream(hw,parameters,data,*_):
            mapper=mapper_class(hw,parameters,0,data,0,0);mapper.make_stream();return{0:mapper.get_stream()}
        hit=self.optimized and not linear and self.cache_key==cache_key
        # Cache is valid only within the same stage, channel group and spatial
        # window. No hash of a reference tensor can authorize reuse across stages.
        p.skipIact=int(hit)
        if self.optimized and not linear:
            from domino_optimized_transport import choose
            module=SimpleNamespace(write_stream=write_stream)
            packet=choose(lambda module,hw,parameters,data,pattern:module.write_stream(hw,parameters,data,0,0),
                          module,self.hardware,p,content,pattern)[0]
        else:packet=write_stream(self.hardware,p,content)[0]
        self.cache_key=cache_key if not linear else None
        if linear:coords=[(0,0,f)for f in(0,4,1,5,2,6,3,7)]
        else:
            key=(width,height,filters,kernel,stride)
            if key not in self.layouts:
                zero=np.zeros((filters,p.psum_size_x,p.psum_size_y),dtype=np.int64)
                self.layouts[key]=self.coordinates(self.hardware,p,zero,io.StringIO())
            coords=self.layouts[key]
        sections=[list(map(int,section))for section in packet]
        return sections,coords,p
