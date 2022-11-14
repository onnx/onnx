# SPDX-License-Identifier: Apache-2.0
# pylint: disable=E1123,W0221

import numpy as np

from ._op import OpRunReduceNumpy

from onnx.defs import onnx_opset_version

class ReduceMin_1_11_12(OpRunReduceNumpy):
    def _run(self, data, axes=None, keepdims=None):  # type: ignore
        axes = tuple(axes) if axes else None
        return (np.minimum.reduce(data, axis=axes, keepdims=keepdims == 1),)


class ReduceMin_18(OpRunReduceNumpy):
    def run(self, data, axes=None):  # type: ignore
        return self._run(data, axes)

    def _run(self, data, axes):  # type: ignore
        if self.IsAxesEmpty(axes) and self.noop_with_empty_axes != 0:
            return (data,)

        axes = self.HandleAxes(axes)
        keepdims = self.keepdims != 0
        return (np.minimum.reduce(data, axis=axes, keepdims=keepdims),)

if onnx_opset_version() >= 18:
    ReduceMin = ReduceMin_18
else:
    ReduceMin = ReduceMin_1_11_12  # type: ignore
