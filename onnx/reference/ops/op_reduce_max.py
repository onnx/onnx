# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
# pylint: disable=E1123,W0221

import numpy as np

from onnx.reference.ops._op import OpRunReduceNumpy


class ReduceMax_1(OpRunReduceNumpy):
    def _run(self, data, axes=None, keepdims=None):  # type: ignore
        axes = tuple(axes) if axes is not None else None
        if data.size == 0:
            output_shape = self.output_shape(data, axes, keepdims).shape
            minvalue = (
                np.iinfo(data.dtype).min
                if np.issubdtype(data.dtype, np.integer)
                else -np.inf
            )
            res = np.full(output_shape, minvalue, dtype=data.dtype)
        else:
            res = np.maximum.reduce(data, axis=axes, keepdims=keepdims == 1)
        if keepdims == 0 and not isinstance(res, np.ndarray):
            # The runtime must return a numpy array of a single float.
            res = np.array(res)
        return (res,)


class ReduceMax_18(OpRunReduceNumpy):
    def _run(self, data, axes=None, keepdims: int = 1, noop_with_empty_axes: int = 0):  # type: ignore
        if self.is_axes_empty(axes) and noop_with_empty_axes != 0:  # type: ignore
            return (data,)

        axes = self.handle_axes(axes)
        keepdims = keepdims != 0  # type: ignore
        res = np.maximum.reduce(data, axis=axes, keepdims=keepdims)
        if keepdims == 0 and not isinstance(res, np.ndarray):
            # The runtime must return a numpy array of a single float.
            res = np.array(res)
        return (res,)
