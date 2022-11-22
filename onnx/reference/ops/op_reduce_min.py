# SPDX-License-Identifier: Apache-2.0
# pylint: disable=E1123,W0221

import numpy as np

from onnx.defs import onnx_opset_version

from ._op import OpRunReduceNumpy


class ReduceMin_1(OpRunReduceNumpy):
    def _run(self, data, axes=None, keepdims=None):  # type: ignore
        axes = tuple(axes) if axes else None
        return (np.minimum.reduce(data, axis=axes, keepdims=keepdims == 1),)


class ReduceMin_11(ReduceMin_1):
    pass


class ReduceMin_12(ReduceMin_1):
    pass


class ReduceMin_13(ReduceMin_1):
    pass


class ReduceMin_18(OpRunReduceNumpy):
    def run(self, data, axes=None, keepdims=None, noop_with_empty_axes=None):  # type: ignore
        keepdims = keepdims or self.keepdims  # type: ignore
        noop_with_empty_axes = noop_with_empty_axes or self.noop_with_empty_axes  # type: ignore
        return self._run(data, axes, keepdims, noop_with_empty_axes)

    def _run(self, data, axes, keepdims=1, noop_with_empty_axes=0):  # type: ignore
        if self.is_axes_empty(axes) and noop_with_empty_axes != 0:  # type: ignore
            return (data,)

        axes = self.handle_axes(axes)
        keepdims = keepdims != 0  # type: ignore
        return (np.minimum.reduce(data, axis=axes, keepdims=keepdims),)


if onnx_opset_version() >= 18:
    ReduceMin = ReduceMin_18
elif onnx_opset_version() >= 13:
    ReduceMin = ReduceMin_13  # type: ignore
elif onnx_opset_version() >= 12:
    ReduceMin = ReduceMin_12  # type: ignore
elif onnx_opset_version() >= 11:
    ReduceMin = ReduceMin_11  # type: ignore
else:
    ReduceMin = ReduceMin_1  # type: ignore
