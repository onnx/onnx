# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy as np

from onnx.defs import onnx_opset_version

from ._op import OpRunReduceNumpy


class ReduceL2_1_11_13(OpRunReduceNumpy):
    def _run(self, data, axes=None, keepdims=None):  # type: ignore
        return (
            np.sqrt(np.sum(np.square(data), axis=axes, keepdims=keepdims)).astype(
                dtype=data.dtype
            ),
        )


class ReduceL2_18(OpRunReduceNumpy):
    def run(self, data, axes=None):  # type: ignore
        return self._run(data, axes)

    def _run(self, data, axes):  # type: ignore
        if self.is_axes_empty(axes) and self.noop_with_empty_axes:
            return (data,)

        axes = self.HandleAxes(axes)
        keepdims = self.keepdims != 0
        return (
            np.sqrt(np.sum(np.square(data), axis=axes, keepdims=keepdims)).astype(
                dtype=data.dtype
            ),
        )


if onnx_opset_version() >= 18:
    ReduceL2 = ReduceL2_18
else:
    ReduceL2 = ReduceL2_1_11_13  # type: ignore
