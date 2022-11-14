# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy as np

from onnx.defs import onnx_opset_version

from ._op import OpRunReduceNumpy


class ReduceL1_1_11_13(OpRunReduceNumpy):
    def _run(self, data, axes=None, keepdims=None):  # type: ignore
        return (
            np.sum(np.abs(data), axis=axes, keepdims=keepdims).astype(dtype=data.dtype),
        )


class ReduceL1_18(OpRunReduceNumpy):
    def run(self, data, axes=None):  # type: ignore
        return self._run(data, axes)

    def _run(self, data, axes):  # type: ignore
        if self.IsAxesEmpty(axes) and self.noop_with_empty_axes:
            return (data,)

        axes = self.HandleAxes(axes)
        keepdims = self.keepdims != 0
        return (np.sum(np.abs(data), axis=axes, keepdims=keepdims).astype(dtype=data.dtype),)


if onnx_opset_version() >= 18:
    ReduceL1 = ReduceL1_18
else:
    ReduceL1 = ReduceL1_1_11_13  # type: ignore
