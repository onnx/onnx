# SPDX-License-Identifier: Apache-2.0
# pylint: disable=E1123,W0221

import numpy as np

from onnx.defs import onnx_opset_version

from ._op import OpRunReduceNumpy


class ReduceMax_1_11_12(OpRunReduceNumpy):
    def _run(self, data, axes=None, keepdims=None):  # type: ignore
        axes = tuple(axes) if axes else None
        return (np.maximum.reduce(data, axis=axes, keepdims=keepdims == 1),)


class ReduceMax_18(OpRunReduceNumpy):
    def run(self, data, axes=None):  # type: ignore
        return self._run(data, axes)

    def _run(self, data, axes):  # type: ignore
        if self.is_axes_empty(axes) and self.noop_with_empty_axes != 0:
            return (data,)

        axes = self.HandleAxes(axes)
        keepdims = self.keepdims != 0
        return (np.maximum.reduce(data, axis=axes, keepdims=keepdims),)


if onnx_opset_version() >= 18:
    ReduceMax = ReduceMax_18
else:
    ReduceMax = ReduceMax_1_11_12  # type: ignore
