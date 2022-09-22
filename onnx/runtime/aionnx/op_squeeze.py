# SPDX-License-Identifier: Apache-2.0
# pylint: disable=E0203,W0221

import numpy as np  # type: ignore

from ...defs import onnx_opset_version
from ..op_run import OpRun
from ._op import OpRunUnaryNum


class Squeeze_1(OpRunUnaryNum):
    def __init__(self, onnx_node, run_params):  # type: ignore
        OpRunUnaryNum.__init__(self, onnx_node, run_params)
        if isinstance(self.axes, np.ndarray):  # type: ignore
            self.axes = tuple(self.axes)  # type: ignore
        elif self.axes in [[], tuple()]:
            self.axes = None  # type: ignore
        elif isinstance(self.axes, list):
            self.axes = tuple(self.axes)

    def _run(self, data):  # type: ignore
        # TODO: support overridden attributes.
        if isinstance(self.axes, (tuple, list)):
            sq = data
            for a in reversed(self.axes):
                sq = np.squeeze(sq, axis=a)
        else:
            sq = np.squeeze(data, axis=self.axes)
        return (sq,)


class Squeeze_11(Squeeze_1):
    pass


class Squeeze_13(OpRun):
    def __init__(self, onnx_node, run_params):  # type: ignore
        OpRun.__init__(self, onnx_node, run_params)
        self.axes = None

    def _run(self, data, axes=None):  # type: ignore
        if axes is not None:
            if hasattr(axes, "__iter__"):
                sq = np.squeeze(data, axis=tuple(axes))
            else:
                sq = np.squeeze(data, axis=axes)
        else:
            sq = np.squeeze(data)
        return (sq,)


if onnx_opset_version() >= 13:
    Squeeze = Squeeze_13
elif onnx_opset_version() >= 11:
    Squeeze = Squeeze_11  # type: ignore
else:
    Squeeze = Squeeze_1  # type: ignore
