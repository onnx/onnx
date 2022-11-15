# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy as np

from onnx.defs import onnx_opset_version

from ._op import OpRunReduceNumpy


class ReduceProd_1(OpRunReduceNumpy):
    def _run(self, data, axes=None, keepdims=None):  # type: ignore
        return (np.prod(data, axis=axes, keepdims=keepdims, dtype=data.dtype),)


class ReduceProd_11(ReduceProd_1):
    pass


class ReduceProd_13(ReduceProd_1):
    pass


class ReduceProd_18(OpRunReduceNumpy):
    def run(self, data, axes=None):  # type: ignore
        return self._run(data, axes)

    def _run(self, data, axes):  # type: ignore
        if self.is_axes_empty(axes) and self.noop_with_empty_axes:  # type: ignore
            return (data,)

        axes = self.HandleAxes(axes)
        keepdims = self.keepdims != 0  # type: ignore
        return (np.prod(data, axis=axes, keepdims=keepdims, dtype=data.dtype),)


if onnx_opset_version() >= 18:
    ReduceProd = ReduceProd_18
elif onnx_opset_version() >= 13:
    ReduceProd = ReduceProd_13  # type: ignore
elif onnx_opset_version() >= 11:
    ReduceProd = ReduceProd_11  # type: ignore
else:
    ReduceProd = ReduceProd_1  # type: ignore
