import json
from pathlib import Path

import numpy as np
import pytest

from fpga.openeye.runtime_activations import (
    activation_record, affine_fixed, check_runtime_tile, convolution,
    residual_add, round_shift_even, validate_activation,
)


def test_activation_shape_hash_and_unsigned_input():
    a = np.arange(24).reshape(2, 4, 3)
    record = activation_record(a, .125)
    assert np.array_equal(validate_activation(record, [2, 4, 3]), a)
    for mutated in [dict(record, sha256='a' * 64), dict(record, scale=0),
                    dict(record, shape=[2, 3, 4]), dict(record, layout='CHW')]:
        with pytest.raises(ValueError):
            validate_activation(mutated)
    with pytest.raises(ValueError):
        activation_record(a.astype(float), .125)
    with pytest.raises(ValueError):
        activation_record(a - 1, .125)


@pytest.mark.parametrize('stride', [1, 2])
def test_reference_matches_scalar_asymmetric_xy(stride):
    a = np.arange(2 * 8 * 6).reshape(2, 8, 6)
    w = (np.arange(2 * 3 * 3 * 3).reshape(2, 3, 3, 3) % 13) - 6
    expected = np.zeros((3, 8 // stride, 6 // stride), dtype=np.int64)
    for f, x, y in np.ndindex(expected.shape):
        for c, ky, kx in np.ndindex((2, 3, 3)):
            ix, iy = x * stride + kx - 1, y * stride + ky - 1
            if 0 <= ix < 8 and 0 <= iy < 6:
                expected[f, x, y] += a[c, ix, iy] * w[c, f, ky, kx]
    assert np.array_equal(convolution(a, w, stride), expected)


@pytest.mark.parametrize('shift', [0, 1, 2, 4, 24])
def test_rounding_matches_python_signed_ties_even(shift):
    a = np.asarray([-17, -10, -6, -3, -2, -1, 0, 1, 2, 3, 6, 10, 17])
    assert np.array_equal(round_shift_even(a, shift), [round(int(v) / 2**shift) for v in a])


def test_affine_signed_bias_relu_saturate_and_overflow():
    a = np.asarray([[[-8, -1, 0, 1, 8, 200]]])
    assert affine_fixed(a, [2], [1], shift=1).tolist() == [[[-8, 0, 0, 2, 8, 127]]]
    assert affine_fixed(a, [2], [1], shift=1, relu=True).tolist() == [[[0, 0, 0, 2, 8, 127]]]
    with pytest.raises(ValueError, match='overflow'):
        affine_fixed(a, [1 << 61], [0])


def test_residual_alignment_add_before_clip():
    a = np.asarray([[[127, -127, -1, 1]]])
    b = np.asarray([[[-127, 127, 0, 0]]])
    # Aligning/clamping the positive branch to 127 before adding is incorrect.
    assert residual_add(a, 2, b, 1, 1).tolist() == [[[127, 0, 0, 2]]]
    with pytest.raises(ValueError):
        residual_add(a, 1, b, 1, 0)
    for shift in (-1, 41, 1.5):
        with pytest.raises(ValueError, match='shift'):
            residual_add(a, 1, b, 1, 1, shift=shift)
    with pytest.raises(ValueError, match='overflow'):
        residual_add(a, 1e20, b, 1, 1)


@pytest.mark.parametrize('relu', [False, True])
def test_host_timing_preserves_signed_rounding_and_clamp(relu):
    values = np.asarray([[[-1024, -17, -3, -1, 0, 1, 3, 17, 1024]],
                         [[1024, 17, 3, 1, 0, -1, -3, -17, -1024]]])
    timings = []
    expected = affine_fixed(values, [-3, 2], [7, -5], shift=2, relu=relu)
    actual = affine_fixed(values, [-3, 2], [7, -5], shift=2, relu=relu, timings=timings)
    assert np.array_equal(actual, expected)
    assert [r['operation'] for r in timings] == [
        'batchnorm_affine_requantization', 'relu_int8_clamp_cast' if relu else 'int8_clamp_cast']
    assert all(r['wall_ns'] >= 0 and r['elements'] == values.size and r['calls'] == 1 for r in timings)
    # Timing metadata must remain scalar and must not contain coefficients/tensors.
    assert all(set(r) == {'operation', 'elements', 'calls', 'wall_ns'} for r in timings)
    a = np.clip(values, -127, 127)
    timings = []
    assert np.array_equal(residual_add(a, 2, -a, 1, .5, timings=timings),
                          residual_add(a, 2, -a, 1, .5))
    assert [r['operation'] for r in timings] == ['residual_align_add_requantization', 'relu_int8_clamp_cast']
    timings = []
    with pytest.raises(ValueError, match='overflow'):
        affine_fixed(values, [1 << 61, 1 << 61], [0, 0], timings=timings)
    assert not timings


def test_wrong_dut_cannot_publish_output(tmp_path):
    tile = {'index': 0, 'filter_start': 0, 'filter_stop': 8,
            'x': {'input_start': 0, 'crop_start': 0},
            'y': {'input_start': 0, 'crop_start': 0}}
    pattern = {'execution': {'input_channels': 2, 'input_width': 4, 'input_height': 4, 'stride': 1},
               'weights': np.ones((2, 8, 3, 3), dtype=int).tolist(),
               'runtime_activations': activation_record(np.ones((2, 4, 4), dtype=int), 1),
               'spatial_tile': tile, 'runtime_output_file': str(tmp_path / 'output.bin'),
               'spatial_report': str(tmp_path / 'report.json')}
    with pytest.raises(AssertionError):
        check_runtime_tile(pattern, np.zeros((8, 4, 4), dtype=int).tolist())
    assert not Path(pattern['runtime_output_file']).exists()
    expected = convolution(pattern['runtime_activations']['values'], pattern['weights'])
    check_runtime_tile(pattern, expected.tolist())
    report = json.loads(Path(pattern['spatial_report']).read_text())
    assert report['checked_values'] == 128
    assert report['activation_sha256'] == pattern['runtime_activations']['sha256']
