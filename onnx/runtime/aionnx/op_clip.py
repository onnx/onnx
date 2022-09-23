# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy as np  # type: ignore

from ...defs import onnx_opset_version
from ..op_run import OpRun
from ._op import OpRunUnaryNum


class Clip_6(OpRunUnaryNum):
    def _run(self, data):  # type: ignore
        amin = getattr(self, "min", None)
        amax = getattr(self, "max", None)
        if amin is None and amax is None:
            return (data,)
        res = np.clip(data, amin, amax)  # type: ignore
        return (res,) if res.dtype == data.dtype else (res.astype(data.dtype),)


class Clip_11(OpRun):
    def _run(self, data, *minmax):  # type: ignore
        le = len(minmax)
        amin = minmax[0] if le > 0 else None
        amax = minmax[1] if le > 1 else None
        if amin is None and amax is None:
            return (data,)
        res = np.clip(data, amin, amax)
        return (res,) if res.dtype == data.dtype else (res.astype(data.dtype),)


if onnx_opset_version() >= 11:
    Clip = Clip_11
else:
    Clip = Clip_6  # type: ignore
