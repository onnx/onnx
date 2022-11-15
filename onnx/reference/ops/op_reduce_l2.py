# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy as np

from onnx.defs import onnx_opset_version

from ._op import OpRunReduceNumpy


class ReduceL2_1(OpRunReduceNumpy):
    def _run(self, data, axes=None, keepdims=None):  # type: ignore
        return (
            np.sqrt(np.sum(np.square(data), axis=axes, keepdims=keepdims)).astype(
                dtype=data.dtype
            ),
        )


class ReduceL2_11(ReduceL2_1):
    pass


class ReduceL2_13(ReduceL2_1):
    pass


class ReduceL2_18(OpRunReduceNumpy):
    def run(self, data, axes=None):  # type: ignore
        return self._run(data, axes)

    def _run(self, data, axes):  # type: ignore
        if self.is_axes_empty(axes) and self.noop_with_empty_axes:  # type: ignore
            return (data,)

        axes = self.HandleAxes(axes)
        keepdims = self.keepdims != 0  # type: ignore
        return (
            np.sqrt(np.sum(np.square(data), axis=axes, keepdims=keepdims)).astype(
                dtype=data.dtype
            ),
        )


if onnx_opset_version() >= 18:
    ReduceL2 = ReduceL2_18
elif onnx_opset_version() >= 13:
    ReduceL2 = ReduceL2_13  # type: ignore
elif onnx_opset_version() >= 11:
    ReduceL2 = ReduceL2_11  # type: ignore
else:
    ReduceL2 = ReduceL2_1  # type: ignore
