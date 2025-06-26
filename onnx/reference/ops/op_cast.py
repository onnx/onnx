# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ml_dtypes
import numpy as np

import onnx
from onnx.reference.op_run import OpRun


def _saturating_cast(x: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Saturating cast for float8 and float4 types.

    This function ensures that values outside the representable range
    of the target dtype are clamped to the maximum or minimum representable
    value of that dtype.
    """
    max_val = ml_dtypes.finfo(dtype).max
    min_val = ml_dtypes.finfo(dtype).min
    max_mask = x > max_val
    min_mask = x < min_val
    x = x.astype(dtype)
    x[max_mask] = max_val
    x[min_mask] = min_val
    return x


def cast_to(x: np.ndarray, to: onnx.TensorProto.DataType, saturate: bool):
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
        return _saturating_cast(x, dtype)

    return x.astype(dtype)


class Cast_1(OpRun):
    def _run(self, x, to=None):
        return (cast_to(x, to, saturate=True),)


class Cast_19(OpRun):
    def _run(self, x, to=None, saturate: bool = True):
        return (cast_to(x, to, saturate),)
