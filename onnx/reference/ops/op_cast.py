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
