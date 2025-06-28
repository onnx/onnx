# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

import onnx
from onnx.reference.op_run import OpRun


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
        return onnx.numpy_helper.saturating_cast(x, dtype)

    return x.astype(dtype)


class Cast_1(OpRun):
    def _run(self, x, to=None):
        return (cast_to(x, to, saturate=True),)


class Cast_19(OpRun):
    def _run(self, x, to=None, saturate: bool = True):
        return (cast_to(x, to, saturate),)
