"""Opt-in CXY activation transport and independent integer convolution oracle.

No model calibration/accuracy claim. Tensor files are private run inputs or
intermediate outputs, never uploaded as Actions evidence.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np


def measured_host_call(records, operation, elements, function, *args, **kwargs):
    """Time one actual host call; never convert wall time into FPGA cycles."""
    if records is None:
        return function(*args, **kwargs)
    started = time.perf_counter_ns()
    result = function(*args, **kwargs)
    elapsed = time.perf_counter_ns() - started
    records.append(dict(operation=operation, elements=int(elements), calls=1,
                        wall_ns=elapsed))
    return result


def integer_array(values, shape=None, low=-127, high=127):
    array = np.asarray(values)
    if array.dtype.kind not in "iu" or (shape is not None and list(array.shape) != list(shape)):
        raise ValueError("Integer tensor has the wrong type or shape")
    if not array.size or array.min() < low or array.max() > high:
        raise ValueError("Integer tensor is empty or outside the allowed range")
    return array.astype(np.int64)


def activation_record(values, scale):
    array = integer_array(values, low=0)
    if array.ndim != 3 or not np.isfinite(scale) or scale <= 0:
        raise ValueError("Pilot requires nonnegative CXY INT8 and positive scale")
    return {"schema_version": 1, "layout": "CXY", "dtype": "int8",
            "shape": list(array.shape), "scale": float(scale),
            "sha256": hashlib.sha256(array.astype('i1').tobytes()).hexdigest(),
            "values": array.tolist()}


def validate_activation(record, shape=None):
    if (record.get("schema_version"), record.get("layout"), record.get("dtype")) != (1, "CXY", "int8"):
        raise ValueError("Unsupported runtime activation schema")
    array = integer_array(record["values"], record["shape"], low=0)
    if shape is not None and list(array.shape) != list(shape):
        raise ValueError("Runtime activation geometry mismatch")
    verified = activation_record(array, record["scale"])
    if verified["sha256"] != record["sha256"]:
        raise ValueError("Runtime activation checksum mismatch")
    return array


def convolution(activations, weights, stride=1):
    """CXY input, IOHW weight; exact int64 correlation, same zero padding."""
    a = integer_array(activations, low=0)
    w = integer_array(weights)
    if a.ndim != 3 or w.ndim != 4 or a.shape[0] != w.shape[0]:
        raise ValueError("Convolution shape mismatch")
    kh, kw = w.shape[2:]
    if stride not in (1, 2) or kh % 2 != 1 or kw % 2 != 1:
        raise ValueError("Unsupported convolution kernel/stride")
    padded = np.pad(a, ((0, 0), (kw // 2, kw // 2), (kh // 2, kh // 2)))
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kw, kh), axis=(1, 2))
    # Window axes are kernel_x/kernel_y; OpenEye weights are kernel_y/kernel_x.
    result = np.einsum('cxyij,cfji->fxy', windows[:, ::stride, ::stride], w,
                       dtype=np.int64, optimize=True)
    if result.min() < -(1 << 31) or result.max() >= (1 << 31):
        raise ValueError("Convolution exceeds signed INT32 accumulator")
    return result


def check_runtime_tile(pattern, decoded):
    execution = pattern["execution"]
    linear = pattern.get("layer_type") == "linear"
    shape = ([execution["input_features"], 1, 1] if linear else
             [execution["input_channels"], execution["input_width"], execution["input_height"]])
    a = validate_activation(pattern["runtime_activations"], shape)
    tile = pattern.get("runtime_tile", pattern.get("spatial_tile"))
    bounds = pattern.get("spatial_tile")
    if not linear and (bounds["x"]["input_start"] or bounds["y"]["input_start"] or bounds["x"]["crop_start"] or bounds["y"]["crop_start"]):
        raise ValueError("Runtime pilot supports full-spatial filter blocks only")
    reference = (dense(a[:, 0, 0], pattern["weights"]) if linear else
                 convolution(a, pattern["weights"], execution["stride"]))
    actual = integer_array(decoded, reference.shape, -(1 << 31), (1 << 31) - 1)
    if not np.array_equal(actual, reference):
        index = tuple(np.argwhere(actual != reference)[0])
        raise AssertionError(f"Runtime convolution mismatch at {index}: DUT={actual[index]} ref={reference[index]}")
    data = actual.astype('<i4').tobytes()
    digest = hashlib.sha256(data).hexdigest()
    with Path(pattern["runtime_output_file"]).open('xb') as stream:
        stream.write(data)
    report = {"tile": tile, "checked_values": int(actual.size), "result": "PASS",
              "output_sha256": digest, "reference_sha256": digest,
              "activation_sha256": pattern["runtime_activations"]["sha256"]}
    with Path(pattern["spatial_report"]).open('x') as stream:
        json.dump(report, stream, sort_keys=True)


def dense(activations, weights):
    a, w = integer_array(activations, low=0), integer_array(weights)
    if a.ndim != 1 or w.ndim != 2 or w.shape[1] != a.size:
        raise ValueError("Dense OI geometry mismatch")
    result = w @ a
    if result.min() < -(1 << 31) or result.max() >= (1 << 31):
        raise ValueError("Dense accumulator exceeds INT32")
    return result


def global_average_pool(values):
    a = integer_array(values, low=0)
    if a.ndim != 3:
        raise ValueError("GAP requires CXY input")
    divisor = a.shape[1] * a.shape[2]
    q, r = np.divmod(a.sum(axis=(1, 2)), divisor)
    return q + ((2 * r > divisor) | ((2 * r == divisor) & ((q & 1) != 0)))


def round_shift_even(values, shift):
    """Signed division by 2**shift; nearest with ties to even, no wrap."""
    if not 0 <= shift <= 40:
        raise ValueError("Invalid fixed-point shift")
    a = integer_array(values, low=-(1 << 63) + 1, high=(1 << 63) - 1)
    if shift == 0:
        return a.copy()
    q, r = np.divmod(np.abs(a), 1 << shift)
    half = 1 << (shift - 1)
    q += (r > half) | ((r == half) & ((q & 1) != 0))
    return np.where(a < 0, -q, q)


def affine_fixed(values, multiplier, bias, shift=24, relu=False, timings=None):
    a = integer_array(values, low=-(1 << 31), high=(1 << 31) - 1)
    m = integer_array(multiplier, [a.shape[0]], -(1 << 62), (1 << 62) - 1)
    b = integer_array(bias, [a.shape[0]], -(1 << 62), (1 << 62) - 1)
    # Check before arithmetic, including negative coefficients and offsets.
    bound = int(np.abs(a).max()) * int(np.abs(m).max()) + int(np.abs(b).max())
    if bound >= (1 << 63):
        raise ValueError("Fixed-point affine would overflow INT64")
    if timings is None:
        q = round_shift_even(a * m[:, None, None] + b[:, None, None], shift)
        return np.clip(q, 0 if relu else -127, 127).astype(np.int64)
    # Fixed-point BN coefficients already incorporate scale conversion. The
    # affine timer is multiply+bias, not standalone floating-point BatchNorm.
    wide = measured_host_call(timings, 'batchnorm_affine_int64', a.size,
        lambda: a * m[:, None, None] + b[:, None, None])
    q = measured_host_call(timings, 'requantization_round', a.size, round_shift_even, wide, shift)
    if relu:
        q = measured_host_call(timings, 'relu', a.size, np.maximum, q, 0)
    q = measured_host_call(timings, 'int8_saturation', a.size, np.clip, q, -127, 127)
    return measured_host_call(timings, 'int64_output_copy', a.size, q.astype, np.int64)


def residual_add(main, main_scale, skip, skip_scale, output_scale, shift=24, timings=None):
    if not isinstance(shift, int) or not 0 <= shift <= 40:
        raise ValueError("Invalid fixed-point shift")
    a = integer_array(main)
    b = integer_array(skip, a.shape)
    scales = np.asarray([main_scale, skip_scale, output_scale], dtype=np.float64)
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("Residual scales must be finite and positive")
    ratios = scales[:2] / output_scale * (1 << shift)
    if not np.all(np.isfinite(ratios)) or np.any(ratios >= (1 << 63) / 254):
        raise ValueError("Residual alignment would overflow INT64")
    coefficients = [int(round(s)) for s in ratios]
    if max(coefficients) * 254 >= (1 << 63):
        raise ValueError("Residual alignment would overflow INT64")
    # Add in a wide accumulator and round once; never clamp branches first.
    if timings is None:
        q = round_shift_even(a * coefficients[0] + b * coefficients[1], shift)
        return np.clip(q, 0, 127).astype(np.int64)
    left, right = measured_host_call(timings, 'residual_scale_alignment', a.size,
        lambda: (a * coefficients[0], b * coefficients[1]))
    wide = measured_host_call(timings, 'residual_add', a.size, np.add, left, right)
    q = measured_host_call(timings, 'requantization_round', a.size, round_shift_even, wide, shift)
    q = measured_host_call(timings, 'relu', a.size, np.maximum, q, 0)
    q = measured_host_call(timings, 'int8_saturation', a.size, np.clip, q, -127, 127)
    return measured_host_call(timings, 'int64_output_copy', a.size, q.astype, np.int64)
