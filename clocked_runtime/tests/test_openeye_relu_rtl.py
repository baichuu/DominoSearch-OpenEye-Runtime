"""Real OpenEye activation-block simulation; no convolution/network claim."""
from pathlib import Path
import os
import shutil

import numpy as np
import pytest

from fpga.openeye.rtl_activation import ReLUBackend


@pytest.fixture
def openeye():
    path=Path(os.environ.get('OPENEYE_NEW_PIN_ROOT',Path.home()/'Projects/OpenEye-9a742fb'))
    if not path.exists()or not shutil.which('iverilog'):pytest.skip('New-pin OpenEye + Icarus required')
    return path


def test_upstream_unsigned_relu_failure_is_reproduced(openeye,tmp_path):
    backend=ReLUBackend(openeye,tmp_path/'private',tmp_path/'evidence',width=8,fix_signed=False)
    with pytest.raises(AssertionError,match='independent signed golden'):
        backend.run(np.asarray([-128,-127,-1,0,1,127]),'unsigned-negative-control')
    assert not list((tmp_path/'evidence').glob('junit-*'))


def test_exhaustive_signed_int8_relu_and_bypass(openeye,tmp_path):
    backend=ReLUBackend(openeye,tmp_path/'private',tmp_path/'evidence',width=8)
    a=np.arange(-128,128,dtype=np.int64)
    assert np.array_equal(backend.run(a,'int8-exhaustive'),np.maximum(a,0))
    assert np.array_equal(backend.run(a,'bypass',enabled=False,bubbles=2),a)
    assert backend.calls[0]['total_cycles']==64
    assert backend.calls[1]['total_cycles']==192


def test_signed64_ragged_shape_zero_and_extreme_values(openeye,tmp_path):
    backend=ReLUBackend(openeye,tmp_path/'private',tmp_path/'evidence')
    rng=np.random.default_rng(37)
    for n in [1,3,4,5,255,512]:
        a=rng.integers(-(1<<40),1<<40,n,dtype=np.int64)
        a[0]=-(1<<63)
        if n>1:a[-1]=(1<<63)-1
        original=a.copy()
        assert np.array_equal(backend.run(a,'size-'+str(n)),np.maximum(a,0))
        assert np.array_equal(a,original)
    assert sum(r['checked_values']for r in backend.calls)==780
    backend.binary.write_bytes(b'changed')
    with pytest.raises(ValueError,match='binary changed'):backend.run(np.asarray([1]),'tamper')
