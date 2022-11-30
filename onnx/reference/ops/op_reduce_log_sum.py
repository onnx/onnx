# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy as np

from onnx.defs import onnx_opset_version

from ._op import OpRunReduceNumpy


class ReduceLogSum_1(OpRunReduceNumpy):
    def _run(self, data, axes=None, keepdims=True):  # type: ignore
        tax = tuple(axes) if axes else None
        res = np.sum(data, axis=tax, keepdims=keepdims)
        if len(res.shape) > 0:
            return (np.log(res, out=res),)
        return (np.log(res),)


class ReduceLogSum_18(OpRunReduceNumpy):
    def _run(self, data, axes=None, keepdims=1, noop_with_empty_axes=0):  # type: ignore
        if self.is_axes_empty(axes) and noop_with_empty_axes:  # type: ignore
            return (data,)

        axes = self.handle_axes(axes)
        keepdims = keepdims != 0  # type: ignore

        res = np.sum(data, axis=axes, keepdims=keepdims)
        if len(res.shape) > 0:
            return (np.log(res, out=res),)
        return (np.log(res),)


if onnx_opset_version() >= 18:
    ReduceLogSum = ReduceLogSum_18
else:
    ReduceLogSum = ReduceLogSum_1  # type: ignore
