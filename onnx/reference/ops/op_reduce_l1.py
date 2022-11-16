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


class ReduceL1_11(ReduceL1_1):
    pass


class ReduceL1_13(ReduceL1_1):
    pass


class ReduceL1_18(OpRunReduceNumpy):
    def run(self, data, axes=None, keepdims=None, noop_with_empty_axes=None):  # type: ignore
        keepdims = keepdims or self.keepdims  # type: ignore
        noop_with_empty_axes = noop_with_empty_axes or self.noop_with_empty_axes  # type: ignore
        return self._run(data, axes, keepdims, noop_with_empty_axes)

    def _run(self, data, axes, keepdims=1, noop_with_empty_axes=0):  # type: ignore
        if self.is_axes_empty(axes) and noop_with_empty_axes:  # type: ignore
            return (data,)

        axes = self.handle_axes(axes)
        keepdims = keepdims != 0  # type: ignore
        return (
            np.sum(np.abs(data), axis=axes, keepdims=keepdims).astype(dtype=data.dtype),
        )


if onnx_opset_version() >= 18:
    ReduceL1 = ReduceL1_18
elif onnx_opset_version() >= 13:
    ReduceL1 = ReduceL1_13  # type: ignore
elif onnx_opset_version() >= 11:
    ReduceL1 = ReduceL1_11  # type: ignore
else:
    ReduceL1 = ReduceL1_1  # type: ignore
