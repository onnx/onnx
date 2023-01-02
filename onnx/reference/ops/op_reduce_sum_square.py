# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy as np

from onnx.defs import onnx_opset_version

from ._op import OpRunReduceNumpy


class ReduceSumSquare_1(OpRunReduceNumpy):
    def _run(self, data, axes=None, keepdims=None):  # type: ignore
        return (np.sum(np.square(data), axis=axes, keepdims=keepdims),)


class ReduceSumSquare_18(OpRunReduceNumpy):
    def _run(self, data, axes=None, keepdims=1, noop_with_empty_axes=0):  # type: ignore
        if self.is_axes_empty(axes) and noop_with_empty_axes != 0:  # type: ignore
            return (np.square(data),)

        axes = self.handle_axes(axes)
        keepdims = keepdims != 0  # type: ignore
        return (np.sum(np.square(data), axis=axes, keepdims=keepdims),)


if onnx_opset_version() >= 18:
    ReduceSumSquare = ReduceSumSquare_18
else:
    ReduceSumSquare = ReduceSumSquare_1  # type: ignore
