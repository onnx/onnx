# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy as np

from onnx.defs import onnx_opset_version

from ._op import OpRunReduceNumpy


class ReduceSumSquare_1(OpRunReduceNumpy):
    def _run(self, data, axes=None, keepdims=None):  # type: ignore
        return (np.sum(np.square(data), axis=axes, keepdims=keepdims),)


class ReduceSumSquare_11(ReduceSumSquare_1):
    pass


class ReduceSumSquare_13(ReduceSumSquare_1):
    pass


class ReduceSumSquare_18(OpRunReduceNumpy):
    def run(self, data, axes=None):  # type: ignore
        return self._run(data, axes)

    def _run(self, data, axes):  # type: ignore
        if self.is_axes_empty(axes) and self.noop_with_empty_axes != 0:  # type: ignore
            return (np.square(data),)

        axes = self.HandleAxes(axes)
        keepdims = self.keepdims != 0  # type: ignore
        return (np.sum(np.square(data), axis=axes, keepdims=keepdims),)


if onnx_opset_version() >= 18:
    ReduceSumSquare = ReduceSumSquare_18
elif onnx_opset_version() >= 13:
    ReduceSumSquare = ReduceSumSquare_13  # type: ignore
elif onnx_opset_version() >= 11:
    ReduceSumSquare = ReduceSumSquare_11  # type: ignore
else:
    ReduceSumSquare = ReduceSumSquare_1  # type: ignore
