# SPDX-License-Identifier: Apache-2.0
# pylint: disable=E0203,W0221

import numpy as np  # type: ignore

from ...defs import onnx_opset_version
from ..op_run import OpRun
from ._op import OpRunUnaryNum


class Unsqueeze_1(OpRunUnaryNum):
    def __init__(self, onnx_node, run_params):  # type: ignore
        OpRunUnaryNum.__init__(self, onnx_node, run_params)
        if isinstance(self.axes, np.ndarray):  # type: ignore
            self.axes = tuple(self.axes)  # type: ignore
        elif self.axes in [[], tuple()]:
            self.axes = None  # type: ignore
        elif isinstance(self.axes, list):
            self.axes = tuple(self.axes)  # type: ignore

    def _run(self, data):  # type: ignore
        # TODO: support overridden attributes.
        if isinstance(self.axes, (tuple, list)):  # type: ignore
            sq = data
            for a in self.axes:  # type: ignore
                sq = np.expand_dims(sq, axis=a)
        else:
            raise RuntimeError(
                "axes cannot be None for operator Unsqueeze (Unsqueeze_1)."
            )
        return (sq,)


class Unsqueeze_11(Unsqueeze_1):
    pass


class Unsqueeze_13(OpRun):
    def __init__(self, onnx_node, run_params):  # type: ignore
        OpRun.__init__(self, onnx_node, run_params)
        self.axes = None

    def _run(self, data, axes=None):  # type: ignore
        if axes is not None:
            if hasattr(axes, "__iter__") and len(axes.shape) > 0:
                sq = np.expand_dims(data, axis=tuple(axes))
            else:
                sq = np.expand_dims(data, axis=axes)
        else:
            raise RuntimeError(
                "axes cannot be None for operator Unsqueeze (Unsqueeze_13)."
            )
        return (sq,)


if onnx_opset_version() >= 13:
    Unsqueeze = Unsqueeze_13
elif onnx_opset_version() >= 11:
    Unsqueeze = Unsqueeze_11  # type: ignore
else:
    Unsqueeze = Unsqueeze_1  # type: ignore
