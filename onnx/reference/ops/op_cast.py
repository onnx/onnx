# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

import onnx
from onnx.reference.op_run import OpRun


def cast_to(
    x: np.ndarray, to: onnx.TensorProto.DataType, saturate: bool, round_mode: str = "up"
):
    if to == onnx.TensorProto.STRING:
        return x.astype(np.str_)

    dtype = onnx.helper.tensor_dtype_to_np_dtype(to)
    if (
        to
        in {
            onnx.TensorProto.FLOAT8E4M3FN,
            onnx.TensorProto.FLOAT8E4M3FNUZ,
            onnx.TensorProto.FLOAT8E5M2,
            onnx.TensorProto.FLOAT8E5M2FNUZ,
        }
        and saturate
    ):
        return onnx.numpy_helper.saturate_cast(x, dtype)

    if to == onnx.TensorProto.FLOAT8E8M0:
        return onnx.numpy_helper.to_float8e8m0(x, saturate, round_mode).astype(dtype)

    if to == onnx.TensorProto.FLOAT6E2M3:
        return float32_to_float6e2m3(x.astype(np.float32), saturate)
    if to == onnx.TensorProto.FLOAT6E3M2:
        return float32_to_float6e3m2(x.astype(np.float32), saturate)

    return x.astype(dtype)


class Cast_1(OpRun):
    def _run(self, x, to=None):
        return (cast_to(x, to, saturate=True, round_mode="up"),)


class Cast_19(OpRun):
    def _run(self, x, to=None, saturate=None):
        return (cast_to(x, to, saturate, round_mode="up"),)


class Cast_24(OpRun):
    def _run(self, x, to=None, saturate=None, round_mode=None):
        return (cast_to(x, to, saturate, round_mode),)


def float32_to_float6e2m3(x: np.ndarray, saturate: bool) -> np.ndarray:
    x = x.astype(np.float32)
    sign_bit = (np.signbit(x).astype(np.uint8) << 5)
    abs_x = np.abs(x)
    # Handle zeros early
    is_zero = abs_x == 0
    # Constants
    bias = 3
    n_mant = 3
    max_exp = 3
    max_mant = (1 << n_mant) - 1

    # Compute unbiased exponent for nonzeros
    e = np.floor(np.log2(np.where(is_zero, 1.0, abs_x))).astype(np.int32)
    exp_biased = e + bias

    # Normalized where exp_biased in [1, max_exp]
    is_normal = (exp_biased >= 1) & (exp_biased <= max_exp)
    frac = abs_x / (2.0 ** e) - 1.0
    mant_f = frac * (1 << n_mant)
    mant_r = np.round(mant_f).astype(np.int32)  # ties to even

    # Handle carry from rounding
    carry = mant_r > max_mant
    mant_r = np.where(carry, 0, mant_r)
    exp_biased = np.where(carry, exp_biased + 1, exp_biased)

    # Subnormal where exp_biased <= 0
    is_sub = exp_biased <= 0
    sub_scale = abs_x / (2.0 ** (1 - bias))
    sub_mant_f = sub_scale * (1 << n_mant)
    sub_mant_r = np.round(sub_mant_f).astype(np.int32)
    # Promote to smallest normal if rounded up to 2^n
    promote = sub_mant_r > max_mant
    sub_mant_r = np.where(promote, 0, sub_mant_r)
    exp_biased = np.where(is_sub & promote, 1, exp_biased)

    # Compose mantissa and exponent
    final_exp = np.clip(exp_biased, 0, max_exp)
    final_mant = np.where(is_sub, sub_mant_r, mant_r)
    final_mant = np.clip(final_mant, 0, max_mant)

    # Saturation for overflow
    overflow = final_exp > max_exp
    if saturate:
        pos_sat = (0 << 5) | (max_exp << 3) | max_mant
        neg_sat = (1 << 5) | (max_exp << 3) | max_mant
        sat_val = np.where(sign_bit != 0, neg_sat, pos_sat).astype(np.uint8)
    else:
        # Non-saturating path: map to max finite as well (no NaN encoding available)
        pos_sat = (0 << 5) | (max_exp << 3) | max_mant
        neg_sat = (1 << 5) | (max_exp << 3) | max_mant
        sat_val = np.where(sign_bit != 0, neg_sat, pos_sat).astype(np.uint8)

    # Base encode
    base = (final_exp.astype(np.uint8) << 3) | (final_mant.astype(np.uint8))
    out = (sign_bit | base).astype(np.uint8)

    # Handle specials
    is_nan_or_inf = np.isnan(x) | np.isinf(x)
    out = np.where(overflow, sat_val, out)
    out = np.where(is_nan_or_inf, sat_val, out)
    # Force -0 to +0
    out = np.where(is_zero, 0, out)
    return out

def float32_to_float6e3m2(x: np.ndarray, saturate: bool) -> np.ndarray:
    x = x.astype(np.float32)
    sign_bit = (np.signbit(x).astype(np.uint8) << 5)
    abs_x = np.abs(x)
    is_zero = abs_x == 0
    bias = 4
    n_mant = 2
    max_exp = 7
    max_mant = (1 << n_mant) - 1

    e = np.floor(np.log2(np.where(is_zero, 1.0, abs_x))).astype(np.int32)
    exp_biased = e + bias

    is_normal = (exp_biased >= 1) & (exp_biased <= max_exp)
    frac = abs_x / (2.0 ** e) - 1.0
    mant_f = frac * (1 << n_mant)
    mant_r = np.round(mant_f).astype(np.int32)

    carry = mant_r > max_mant
    mant_r = np.where(carry, 0, mant_r)
    exp_biased = np.where(carry, exp_biased + 1, exp_biased)

    is_sub = exp_biased <= 0
    sub_scale = abs_x / (2.0 ** (1 - bias))
    sub_mant_f = sub_scale * (1 << n_mant)
    sub_mant_r = np.round(sub_mant_f).astype(np.int32)
    promote = sub_mant_r > max_mant
    sub_mant_r = np.where(promote, 0, sub_mant_r)
    exp_biased = np.where(is_sub & promote, 1, exp_biased)

    final_exp = np.clip(exp_biased, 0, max_exp)
    final_mant = np.where(is_sub, sub_mant_r, mant_r)
    final_mant = np.clip(final_mant, 0, max_mant)

    overflow = final_exp > max_exp
    if saturate:
        pos_sat = (0 << 5) | (max_exp << 2) | max_mant
        neg_sat = (1 << 5) | (max_exp << 2) | max_mant
        sat_val = np.where(sign_bit != 0, neg_sat, pos_sat).astype(np.uint8)
    else:
        pos_sat = (0 << 5) | (max_exp << 2) | max_mant
        neg_sat = (1 << 5) | (max_exp << 2) | max_mant
        sat_val = np.where(sign_bit != 0, neg_sat, pos_sat).astype(np.uint8)

    base = (final_exp.astype(np.uint8) << 2) | (final_mant.astype(np.uint8))
    out = (sign_bit | base).astype(np.uint8)

    is_nan_or_inf = np.isnan(x) | np.isinf(x)
    out = np.where(overflow, sat_val, out)
    out = np.where(is_nan_or_inf, sat_val, out)
    out = np.where(is_zero, 0, out)
    return out
