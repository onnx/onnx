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

    if dtype in {onnx.TensorProto.FLOAT6E2M3, onnx.TensorProto.FLOAT6E3M2}:
        # TODO: Implement FP6 casting with rounding/saturation
        return x.astype(dtype)

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
    sign = np.signbit(x).astype(np.uint8) << 5
    abs_x = np.abs(x)
    is_zero = abs_x == 0
    exp = np.floor(np.log2(np.where(is_zero, 1, abs_x))).astype(np.int8) + 3  # Bias 3 for E2M3
    mant = np.round((abs_x / (2 ** (exp - 3))) * 8).astype(np.uint8) & 0x07
    val = sign | ((np.clip(exp, 0, 3) << 3) | mant)
    val = np.where(is_zero & sign, 0, val)  # -0 to 0
    if saturate:
        val = np.where(abs_x > 24, sign | 0x3F, val)  # Saturate max
        val = np.where(np.isinf(abs_x) | np.isnan(x), sign | 0x3F, val)
    else:
        val = np.where(abs_x > 24 | np.isinf(abs_x) | np.isnan(x), np.nan, val)
    return val.astype(np.uint8)

def float32_to_float6e3m2(x: np.ndarray, saturate: bool) -> np.ndarray:
    # Similar logic with bias 4, mant 2 bits, max 48
    sign = np.signbit(x).astype(np.uint8) << 5
    abs_x = np.abs(x)
    exp = np.floor(np.log2(abs_x + 1e-20)).astype(np.int8) + 4
    mant = np.round((abs_x / (2** (exp - 4))) * 4).astype(np.uint8) & 0x03
    val = sign | ((np.clip(exp, 0, 7) << 2) | mant)
    if saturate:
        val = np.where(abs_x > 48, (sign | 0x3F), val)
    else:
        val = np.where(abs_x > 48, np.nan, val)
    val = np.where(np.isnan(x), (sign | 0x3F) if saturate else np.nan, val)
    return val.astype(np.uint8)
