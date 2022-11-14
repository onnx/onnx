# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy as np

from onnx.helper import float32_to_bfloat16, tensor_dtype_to_np_dtype
from onnx.numpy_helper import bfloat16_to_float32
from onnx.onnx_pb import TensorProto
from onnx.reference.op_run import OpRun

bfloat16 = np.dtype((np.uint16, {"ui": (np.uint16, 0)}))


def cast_to(x, to):
    if x.dtype == bfloat16:
        if to == TensorProto.BFLOAT16:
            return x
        xr = x.ravel()
        xf = np.empty(xr.shape[0], dtype=np.float32)
        for i in range(xr.shape[0]):
            el = bfloat16_to_float32(xr[i])
            xf[i] = el
        dtype = tensor_dtype_to_np_dtype(to)
        return xf.astype(dtype).reshape(x.shape)
    if to == TensorProto.BFLOAT16:
        xf = x.astype(np.float32).ravel()
        y = np.empty(xf.shape, dtype=bfloat16).ravel()
        for i in range(y.shape[0]):
            el = float32_to_bfloat16(xf[i], truncate=True)
            y[i] = el
        return y.reshape(x.shape)

    dtype = tensor_dtype_to_np_dtype(to)
    return x.astype(dtype)


class Cast(OpRun):
    def _run(self, x, to=None):  # type: ignore
        return (cast_to(x, to),)
