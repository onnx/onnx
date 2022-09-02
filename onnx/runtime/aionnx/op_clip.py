# SPDX-License-Identifier: Apache-2.0

import numpy as np

from onnx.defs import onnx_opset_version

from ._op import OpRun, OpRunUnaryNum


class Clip_6(OpRunUnaryNum):
    def __init__(self, onnx_node, log_function):
        OpRunUnaryNum.__init__(self, onnx_node, log_function)

    def _run(self, data, attributes=None):  # noqa: W0221
        res = np.clip(data, getattr(self, "min", None), getattr(self, "max", None))
        return (res,) if res.dtype == data.dtype else (res.astype(data.dtype),)


class Clip_11(OpRun):
    def __init__(self, onnx_node, log_function):
        OpRun.__init__(self, onnx_node, log_function)

    def _run(self, data, *minmax, attributes=None):  # noqa: W0221
        le = len(minmax)
        amin = minmax[0] if le > 0 else None
        amax = minmax[1] if le > 1 else None
        res = np.clip(data, amin, amax)
        return (res,) if res.dtype == data.dtype else (res.astype(data.dtype),)


if onnx_opset_version() >= 11:
    Clip = Clip_11
else:
    Clip = Clip_6
