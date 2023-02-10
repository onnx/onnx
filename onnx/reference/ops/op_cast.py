# SPDX-License-Identifier: Apache-2.0
# pylint: disable=R0912,W0221

import numpy as np

from onnx.helper import (
    float32_to_bfloat16,
    float32_to_floate4m3,
    float32_to_floate5m2,
    tensor_dtype_to_np_dtype,
)
from onnx.mapping import TENSOR_TYPE_MAP
from onnx.numpy_helper import (
    bfloat16_to_float32,
    floate4m3_to_float32,
    floate5m2_to_float32,
)
from onnx.onnx_pb import TensorProto
from onnx.reference.custom_element_types import bfloat16, floate4m3, floate5m2
from onnx.reference.op_run import OpRun


def cast_to(x, to):

    if x.dtype == bfloat16 and x.dtype.descr[0][0] == "bfloat16":
        if to == TensorProto.BFLOAT16:
            return x
        xr = x.ravel()
        xf = np.empty(xr.shape[0], dtype=np.float32)
        for i in range(xr.shape[0]):
            el = bfloat16_to_float32(xr[i])
            xf[i] = el
        dtype = tensor_dtype_to_np_dtype(to)
        return xf.astype(dtype).reshape(x.shape)

    if x.dtype == floate4m3 and x.dtype.descr[0][0] == "e4m3":
        if to == TensorProto.FLOATE4M3:
            return x
        xr = x.ravel()
        xf = np.empty(xr.shape[0], dtype=np.float32)
        for i in range(xr.shape[0]):
            el = floate4m3_to_float32(xr[i])
            xf[i] = el
        dtype = tensor_dtype_to_np_dtype(to)
        return xf.astype(dtype).reshape(x.shape)

    if x.dtype == floate5m2 and x.dtype.descr[0][0] == "e5m2":
        if to == TensorProto.FLOATE5M2:
            return x
        xr = x.ravel()
        xf = np.empty(xr.shape[0], dtype=np.float32)
        for i in range(xr.shape[0]):
            el = floate5m2_to_float32(xr[i])
            xf[i] = el
        dtype = tensor_dtype_to_np_dtype(to)
        return xf.astype(dtype).reshape(x.shape)

    if to == TensorProto.BFLOAT16:
        xf = x.astype(np.float32).ravel()
        y = np.empty(xf.shape, dtype=bfloat16).ravel()
        for i in range(y.shape[0]):
            el = float32_to_bfloat16(xf[i], truncate=True)  # type: ignore[assignment]
            y[i] = el
        return y.reshape(x.shape)

    if to == TensorProto.FLOATE4M3:
        xf = x.astype(np.float32).ravel()
        y = np.empty(xf.shape, dtype=floate4m3).ravel()
        for i in range(y.shape[0]):
            el = float32_to_floate4m3(xf[i])  # type: ignore[assignment]
            y[i] = el
        return y.reshape(x.shape)

    if to == TensorProto.FLOATE5M2:
        xf = x.astype(np.float32).ravel()
        y = np.empty(xf.shape, dtype=floate5m2).ravel()
        for i in range(y.shape[0]):
            el = float32_to_floate5m2(xf[i])  # type: ignore[assignment]
            y[i] = el
        return y.reshape(x.shape)

    if to == TensorProto.STRING:
        return x.astype(np.str_)

    dtype = tensor_dtype_to_np_dtype(to)
    return x.astype(dtype)


class Cast(OpRun):
    def _run(self, x, to=None):  # type: ignore
        return (cast_to(x, to),)
