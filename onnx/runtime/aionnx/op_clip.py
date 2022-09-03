# SPDX-License-Identifier: Apache-2.0

from typing import Type

import numpy as np  # type: ignore

from onnx.defs import onnx_opset_version

from ._op import OpRun, OpRunUnaryNum


class Clip_6(OpRunUnaryNum):
    def __init__(self, onnx_node, log_function):  # type: ignore
        OpRunUnaryNum.__init__(self, onnx_node, log_function)

    def _run(self, data, attributes=None):  # type: ignore # pylint: disable=W0221
        res = np.clip(data, getattr(self, "min", None), getattr(self, "max", None))
        return (res,) if res.dtype == data.dtype else (res.astype(data.dtype),)


class Clip_11(OpRun):
    def __init__(self, onnx_node, log_function):  # type: ignore
        OpRun.__init__(self, onnx_node, log_function)

    def _run(self, data, *minmax, attributes=None):  # type: ignore # pylint: disable=W0221
        le = len(minmax)
        amin = minmax[0] if le > 0 else None
        amax = minmax[1] if le > 1 else None
        res = np.clip(data, amin, amax)
        return (res,) if res.dtype == data.dtype else (res.astype(data.dtype),)


if onnx_opset_version() >= 11:
    Clip: Type[OpRun] = Clip_11  # type: ignore
else:
    Clip: Type[OpRun] = Clip_6  # type: ignore
