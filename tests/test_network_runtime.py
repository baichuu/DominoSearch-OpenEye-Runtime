import copy
import hashlib

import numpy as np
import pytest

from fpga.openeye.network_layer_tiles import assemble, channel_filter_plan, convolution_execution, validate_retained_activation_capacity
from fpga.openeye.runtime_activations import convolution, dense, global_average_pool
from fpga.openeye.run_network_inference import Runtime


def test_channel_filter_plan_covers_full_weights():
    for channels, filters in [(64,128),(128,256),(256,512),(512,512),(512,1000)]:
        plan = channel_filter_plan(channels, filters)
        coverage = set()
        for t in plan:
            region = {(c,f) for c in range(t['channel_start'],t['channel_stop'])
                      for f in range(t['filter_start'],t['filter_stop'])}
            assert not region.intersection(coverage)
            coverage.update(region)
        assert len(coverage) == channels * filters
    for channels, filters in [(63,8),(65,8),(64,7),(0,8)]:
        with pytest.raises(ValueError):
            channel_filter_plan(channels,filters)


@pytest.mark.parametrize('kernel,stride,size', [(3,2,16),(1,2,16),(3,1,7)])
def test_channel_partial_accumulation_before_quantization(kernel,stride,size):
    a = np.fromfunction(lambda c,x,y: (c+3*x+7*y)%128, (128,size,size),dtype=int)
    w = np.fromfunction(lambda c,f,y,x: (c+f+3*y+x)%13-6,(128,16,kernel,kernel),dtype=int)
    plan, reports, partials = channel_filter_plan(128,16), [], []
    for t in plan:
        part = convolution(a[t['channel_start']:t['channel_stop']],
                           w[t['channel_start']:t['channel_stop'], t['filter_start']:t['filter_stop']],stride)
        digest = hashlib.sha256(part.astype('<i4').tobytes()).hexdigest()
        reports.append(dict(tile=t,result='PASS',output_sha256=digest,reference_sha256=digest))
        partials.append(part)
    reference = convolution(a,w,stride)
    assert np.array_equal(assemble(plan,reports,partials,reference.shape), reference)
    with pytest.raises(AssertionError):
        assemble(plan,reports[:-1],partials,reference.shape)
    wrong = copy.deepcopy(reports)
    wrong[0]['output_sha256'] = 'wrong'
    with pytest.raises(AssertionError):
        assemble(plan,wrong,partials,reference.shape)


def test_fc_oi_reference_and_gap_ties_even():
    a = np.asarray([[[0,1]],[[1,2]],[[2,3]],[[3,4]]])
    gap = global_average_pool(a)
    assert gap.tolist() == [0,2,2,4]
    w = np.asarray([[2,-3,4,-5],[-1,2,-3,4]])
    assert dense(gap,w).tolist() == [sum(int(v)*int(q) for v,q in zip(row,gap)) for row in w]
    with pytest.raises(ValueError):
        dense(gap,w.T)
    a = np.arange(49).reshape(1,7,7)
    assert global_average_pool(a).tolist() == [24]


@pytest.mark.parametrize('kernel,stride,size', [(1,2,16),(3,2,16),(3,1,7),(1,2,14)])
def test_native_kernel_stride_decimation_and_odd_border(kernel,stride,size):
    a = np.fromfunction(lambda c,x,y: (3*c+5*x+7*y)%128,(64,size,size),dtype=int)
    w = np.fromfunction(lambda c,f,y,x: (c+7*f+3*y+x)%11-5,(64,8,kernel,kernel),dtype=int)
    execution = dict(kernel_width=kernel,kernel_height=kernel,stride=stride)
    padded_a, native_w, dut_kernel, dut_stride = convolution_execution(a,w,execution)
    assert native_w is w and dut_kernel == kernel and dut_stride == 1
    result = convolution(padded_a,native_w,1)[:, :size, :size][:, ::stride, ::stride]
    assert np.array_equal(result,convolution(a,w,stride))


@pytest.mark.parametrize('used,capacity', [(4,16),(12,16),(15,16)])
def test_retained_activation_capacity(used,capacity):
    validate_retained_activation_capacity(used,capacity)


@pytest.mark.parametrize('used,capacity', [(0,16),(16,16),(16,32),(12,8)])
def test_retained_activation_capacity_rejects_wrap_or_overflow(used,capacity):
    with pytest.raises(ValueError,match='capacity'):
        validate_retained_activation_capacity(used,capacity)


def test_downsample_consumes_original_skip_not_main_output():
    engine = object.__new__(Runtime)
    engine.blocks = []
    original = np.arange(64*8*8,dtype=int).reshape(64,8,8) % 128
    calls = []
    def stage(layer,a,ref_a,scale,bn=None,relu=False,rtl=True):
        calls.append((layer,a.copy(),bn,relu))
        if layer.endswith('downsample.0'):
            assert np.array_equal(a,original)
        out = np.ones((128,4,4),dtype=int) * (2 if layer.endswith('conv2') else 1)
        return out,out.copy(),1
    engine.stage = stage
    a,ref,scale = engine.block('layer2.0',original,original.copy(),1,downsample=True)
    assert np.array_equal(a,ref) and a.shape==(128,4,4)
    assert [c[0] for c in calls]==['layer2.0.conv1','layer2.0.conv2','layer2.0.downsample.0']
    assert calls[0][3] and not calls[1][3] and not calls[2][3]
    assert engine.blocks[0]['skip_shape']==[128,4,4]
