# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.ops._op import OpRunReduceNumpy


def compute_log_sum_exp(data, axes, keepdims):
    data_max = data.copy()
    ind = np.isinf(data_max)
    data_max[ind] = -np.inf
    mx = data_max.max(axis=axes, keepdims=True)
    sub = np.subtract(data, mx)
    exp = np.exp(sub, out=sub)
    mxs = np.sum(exp, axis=axes, keepdims=True, dtype=data.dtype)
    res = np.log(mxs) + mx
    if not keepdims:
        res = np.squeeze(res, axis=axes)
    return (res,)


class ReduceLogSumExp_1(OpRunReduceNumpy):
    def _run(self, data, axes=None, keepdims=None):
        tax = tuple(axes) if axes is not None else None

        if data.size == 0:
            return self.reduce_constant(data, -np.inf, tax, keepdims)
        return compute_log_sum_exp(data, tax, keepdims)


class ReduceLogSumExp_18(OpRunReduceNumpy):
    def _run(self, data, axes=None, keepdims=1, noop_with_empty_axes=0):
        axes = self.handle_axes(axes, noop_with_empty_axes)

        keepdims = keepdims != 0

        if data.size == 0:
            return self.reduce_constant(data, -np.inf, axes, keepdims)

        return compute_log_sum_exp(data, axes, keepdims)


class ReduceLogSumExp_28(ReduceLogSumExp_18):
    def _run(self, data, axes=None, keepdims=1, noop_with_empty_axes=0):
        if np.issubdtype(data.dtype, np.integer):
            raise TypeError(
                f"ReduceLogSumExp does not support integer input (got {data.dtype}). "
                "Integer support was removed in opset 28 because Log and Exp are only "
                "defined for float types. Cast the input to a float type."
            )
        return super()._run(
            data,
            axes=axes,
            keepdims=keepdims,
            noop_with_empty_axes=noop_with_empty_axes,
        )
