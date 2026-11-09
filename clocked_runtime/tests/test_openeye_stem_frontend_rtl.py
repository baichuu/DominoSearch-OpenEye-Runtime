import numpy as np
import pytest
from test_openeye_relu_rtl import openeye
from fpga.openeye.rtl_stem import StemBackend


@pytest.mark.parametrize('shape,filters,k,stride,shift',[
    ((1,1,1),1,1,1,0),((3,5,4),3,3,1,1),((3,7,5),2,7,2,2),((3,8,8),2,7,2,24)])
def test_actual_signed_stem_bn_relu_pool_and_stalls(openeye,tmp_path,shape,filters,k,stride,shift,channel_lanes=1):
    rng=np.random.default_rng(71+k)
    a=rng.integers(-127,128,shape);w=rng.integers(-10,11,(shape[0],filters,k,k))
    m=np.arange(filters,dtype=np.int64)*2-1;b=np.arange(filters,dtype=np.int64)*17
    backend=StemBackend(openeye,tmp_path/'private',tmp_path/'evidence',channel_lanes=channel_lanes)
    actual=backend.run(a,w,m,b,shift,stride=stride,sink_stalls=True)
    raw=np.zeros((filters,(shape[1]+stride-1)//stride,(shape[2]+stride-1)//stride),dtype=np.int64)
    # Scalar independent coordinate oracle, including asymmetric X/Y and padding.
    for f in range(filters):
        for x in range(raw.shape[1]):
            for y in range(raw.shape[2]):
                for c in range(shape[0]):
                    for ky in range(k):
                        for kx in range(k):
                            ix=x*stride+kx-k//2;iy=y*stride+ky-k//2
                            if 0<=ix<shape[1] and 0<=iy<shape[2]:raw[f,x,y]+=int(a[c,ix,iy])*int(w[c,f,ky,kx])
    post=np.empty_like(raw)
    for f in range(filters):
        for index in np.ndindex(raw.shape[1:]):
            z=int(raw[(f,*index)])*int(m[f])+int(b[f]);q,r=divmod(z,1<<shift)
            q+=int(2*r>1<<shift or(2*r==1<<shift and q%2))
            post[(f,*index)]=min(127,max(0,q))
    expected=np.zeros_like(actual)
    for f,x,y in np.ndindex(expected.shape):
        expected[f,x,y]=post[f,max(0,x*2-1):min(post.shape[1],x*2+2),max(0,y*2-1):min(post.shape[2],y*2+2)].max()
    assert np.array_equal(actual,expected)
    assert backend.calls[-1]['whole_network_cycles']is None
    assert backend.calls[-1]['frontend_latency_cycles']>a.size


@pytest.mark.parametrize('shape,filters,k,stride,shift',[
    ((3,1,1),1,1,1,0),((3,5,4),3,3,1,1),((3,7,5),2,7,2,2),
    ((3,8,8),2,7,2,24),((6,5,7),2,3,2,2),((3,3,2),2,7,1,0)])
def test_banked_channel_parallel_stem_matches_independent_coordinate_oracle(
        openeye,tmp_path,shape,filters,k,stride,shift):
    test_actual_signed_stem_bn_relu_pool_and_stalls(
        openeye,tmp_path,shape,filters,k,stride,shift,channel_lanes=3)


def test_parallel_banks_preserve_extremes_across_images_and_reduce_measured_cycles(openeye,tmp_path):
    backends=[StemBackend(openeye,tmp_path/f'private-{n}',tmp_path/f'evidence-{n}',channel_lanes=n)for n in(1,3)]
    rng=np.random.default_rng(2961);w=rng.integers(-127,128,(3,4,7,7),dtype=np.int64)
    for index,a in enumerate([np.zeros((3,12,10),dtype=np.int64),
            np.full((3,12,10),-127,dtype=np.int64),rng.integers(-127,128,(3,12,10),dtype=np.int64)]):
        outputs=[b.run(a,w,[1,-1,3,-3],[0,1,-1,100],shift=12,sink_stalls=bool(index%2),site='image-'+str(index))for b in backends]
        assert np.array_equal(*outputs)
        assert backends[0].calls[-1]['convolution_output_sha256']==backends[1].calls[-1]['convolution_output_sha256']
        assert backends[1].calls[-1]['frontend_latency_cycles']<backends[0].calls[-1]['frontend_latency_cycles']
