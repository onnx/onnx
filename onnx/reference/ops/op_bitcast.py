# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

import onnx
from onnx.reference.op_run import OpRun


class BitCast(OpRun):
    def _run(self, x, to: int):  # type: ignore
        if to == onnx.TensorProto.STRING:
            raise ValueError("BitCast to STRING is not supported")
        if x.dtype == np.str_:
            raise ValueError("BitCast from STRING is not supported")

        target_dtype = onnx.helper.tensor_dtype_to_np_dtype(to)

        if x.dtype.itemsize != np.dtype(target_dtype).itemsize:
            raise ValueError(
                f"BitCast requires input and output types to have the same "
                f"bit-width, but got {x.dtype} ({x.dtype.itemsize * 8} bits) "
                f"and {target_dtype} ({np.dtype(target_dtype).itemsize * 8} bits)"
            )

        result = x.view(target_dtype)
        return (result,)
