# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy as np

from onnx.defs import onnx_opset_version

from ._op import OpRunReduceNumpy


class ReduceL1_1(OpRunReduceNumpy):
    def _run(self, data, axes=None, keepdims=None):  # type: ignore
        return (
            np.sum(np.abs(data), axis=axes, keepdims=keepdims).astype(dtype=data.dtype),
        )


class ReduceL1_18(OpRunReduceNumpy):
    def _run(self, data, axes=None, keepdims=1, noop_with_empty_axes=0):  # type: ignore
        if self.is_axes_empty(axes) and noop_with_empty_axes:  # type: ignore
            return (data,)

        axes = self.handle_axes(axes)
        keepdims = keepdims != 0  # type: ignore
        return (
            np.sum(np.abs(data), axis=axes, keepdims=keepdims).astype(dtype=data.dtype),
        )


if onnx_opset_version() >= 18:
    ReduceL1 = ReduceL1_18
else:
    ReduceL1 = ReduceL1_1  # type: ignore
