# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

import onnx
from onnx.reference.op_run import OpRun


class BitCast(OpRun):
    # Sub-byte types not supported in NumPy reference implementation
    _SUB_BYTE_TYPES: frozenset[int] = frozenset(
        {
            onnx.TensorProto.INT4,
            onnx.TensorProto.UINT4,
            onnx.TensorProto.FLOAT4E2M1,
            onnx.TensorProto.INT2,
            onnx.TensorProto.UINT2,
        }
    )

    def _run(self, x, to: int):  # type: ignore
        if to == onnx.TensorProto.STRING:
            raise ValueError("BitCast to STRING is not supported")
        if x.dtype == np.str_:
            raise ValueError("BitCast from STRING is not supported")
        if to in self._SUB_BYTE_TYPES:
            raise ValueError(
                "BitCast to sub-byte types (< 8 bits) is not supported "
                "in the reference implementation"
            )

        target_dtype = onnx.helper.tensor_dtype_to_np_dtype(to)

        if x.dtype.itemsize != np.dtype(target_dtype).itemsize:
            raise ValueError(
                f"BitCast requires input and output types to have the same "
                f"bit-width, but got {x.dtype} ({x.dtype.itemsize * 8} bits) "
                f"and {target_dtype} ({np.dtype(target_dtype).itemsize * 8} bits)"
            )

        result = x.view(target_dtype)
        return (result,)
