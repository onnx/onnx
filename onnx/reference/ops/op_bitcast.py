# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

import onnx
from onnx.reference.op_run import OpRun

# NumPy stores each element of these types in a full byte, so the bit-width
# cannot be derived from ``dtype.itemsize``.
_SUB_BYTE_BIT_WIDTHS = {
    onnx.TensorProto.INT4: 4,
    onnx.TensorProto.UINT4: 4,
    onnx.TensorProto.FLOAT4E2M1: 4,
    onnx.TensorProto.INT2: 2,
    onnx.TensorProto.UINT2: 2,
    onnx.TensorProto.FLOAT6E2M3: 6,
    onnx.TensorProto.FLOAT6E3M2: 6,
}


def _bit_width(dtype: np.dtype) -> int:
    tensor_dtype = onnx.helper.np_dtype_to_tensor_dtype(dtype)
    return _SUB_BYTE_BIT_WIDTHS.get(tensor_dtype, dtype.itemsize * 8)


class BitCast(OpRun):
    def _run(self, x, to: int):  # type: ignore[override]
        if to == onnx.TensorProto.STRING:
            raise ValueError("BitCast to STRING is not supported")
        if x.dtype == np.str_:
            raise ValueError("BitCast from STRING is not supported")

        target_dtype = np.dtype(onnx.helper.tensor_dtype_to_np_dtype(to))
        input_bits = _bit_width(x.dtype)
        output_bits = _bit_width(target_dtype)

        if input_bits != output_bits:
            raise ValueError(
                f"BitCast requires input and output types to have the same "
                f"bit-width, but got {x.dtype} ({input_bits} bits) "
                f"and {target_dtype} ({output_bits} bits)"
            )

        result = x.view(target_dtype)
        return (result,)
