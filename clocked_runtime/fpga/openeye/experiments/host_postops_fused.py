"""Explicit research backends for bit-exact BN/requantization/ReLU and residuals.

Baseline runtime/oracle remain unchanged. Library compilation is a separate,
recorded operation; importing this module never compiles or selects a backend.
"""
import ctypes
from pathlib import Path
import numpy as np


def checked(values, shape=None, low=-(1 << 31), high=(1 << 31)-1):
    a=np.asarray(values)
    if a.dtype.kind not in 'iu' or not a.size or (shape is not None and list(a.shape)!=list(shape)):
        raise ValueError('Integer tensor has the wrong type or shape')
    if a.min()<low or a.max()>high:
        raise ValueError('Integer tensor is outside allowed range')
    return np.ascontiguousarray(a,dtype=np.int64)


def shift_value(shift):
    if type(shift)is not int or not 0<=shift<=40:
        raise ValueError('Invalid fixed-point shift')
    return shift


def affine_inputs(values,multiplier,bias,shift):
    shift_value(shift)
    a=checked(values)
    if a.ndim!=3:
        raise ValueError('Require CXY affine tensor')
    m=checked(multiplier,[a.shape[0]],-(1<<62),(1<<62)-1)
    b=checked(bias,[a.shape[0]],-(1<<62),(1<<62)-1)
    bound=int(np.abs(a).max())*int(np.abs(m).max())+int(np.abs(b).max())
    if bound>=(1<<63):
        raise ValueError('Fixed-point affine would overflow INT64')
    return a,m,b


def residual_inputs(main,main_scale,skip,skip_scale,output_scale,shift):
    shift_value(shift)
    a=checked(main,low=-127,high=127);b=checked(skip,a.shape,low=-127,high=127)
    scales=np.asarray([main_scale,skip_scale,output_scale],dtype=np.float64)
    if not np.all(np.isfinite(scales)) or np.any(scales<=0):
        raise ValueError('Residual scales must be finite and positive')
    ratios=scales[:2]/output_scale*(1<<shift)
    if not np.all(np.isfinite(ratios)) or np.any(ratios>=(1<<63)/254):
        raise ValueError('Residual alignment would overflow INT64')
    coeff=[int(round(s))for s in ratios]
    if max(coeff)*254>=(1<<63):
        raise ValueError('Residual alignment would overflow INT64')
    return a,b,coeff


def round_clamp_numpy(values,shift,relu):
    # Arithmetic floor quotient and nonnegative remainder also handle negatives.
    # Every operation writes a fresh scratch/output; inputs are never modified.
    if shift:
        q=np.right_shift(values,shift)
        r=np.bitwise_and(values,(1<<shift)-1)
        half=1<<(shift-1)
        increment=(r>half)|((r==half)&((q&1)!=0))
        np.add(q,increment,out=q)
    else:
        q=values.copy()
    np.clip(q,0 if relu else -127,127,out=q)
    return q


def affine_numpy(values,multiplier,bias,shift=24,relu=False):
    a,m,b=affine_inputs(values,multiplier,bias,shift)
    wide=np.multiply(a,m[:,None,None])
    np.add(wide,b[:,None,None],out=wide)
    return round_clamp_numpy(wide,shift,relu)


def residual_numpy(main,main_scale,skip,skip_scale,output_scale,shift=24):
    a,b,c=residual_inputs(main,main_scale,skip,skip_scale,output_scale,shift)
    wide=np.multiply(a,c[0]);np.add(wide,np.multiply(b,c[1]),out=wide)
    return round_clamp_numpy(wide,shift,True)


class Native:
    def __init__(self,library):
        self.path=Path(library).resolve()
        self.library=ctypes.CDLL(str(self.path))
        pointer=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
        self.library.domino_affine.argtypes=[pointer,pointer,pointer,pointer,ctypes.c_size_t,ctypes.c_size_t,ctypes.c_uint,ctypes.c_int]
        self.library.domino_affine.restype=None
        self.library.domino_residual.argtypes=[pointer,pointer,pointer,ctypes.c_size_t,ctypes.c_int64,ctypes.c_int64,ctypes.c_uint]
        self.library.domino_residual.restype=None

    def affine(self,values,multiplier,bias,shift=24,relu=False):
        a,m,b=affine_inputs(values,multiplier,bias,shift);out=np.empty_like(a)
        self.library.domino_affine(a,m,b,out,a.shape[0],a.shape[1]*a.shape[2],shift,int(relu))
        return out

    def residual(self,main,main_scale,skip,skip_scale,output_scale,shift=24):
        a,b,c=residual_inputs(main,main_scale,skip,skip_scale,output_scale,shift);out=np.empty_like(a)
        self.library.domino_residual(a,b,out,a.size,c[0],c[1],shift)
        return out
