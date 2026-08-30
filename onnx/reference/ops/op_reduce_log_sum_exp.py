# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.ops._op import OpRunReduceNumpy


def _cast_integer_to_float(data):
    """Cast integer tensors to float64.

    ``log``/``exp`` are undefined for integer dtypes and the ``-np.inf``
    constant cannot be stored in an integer array, so the reference
    implementation works in floating point for integer inputs.
    """
    if np.issubdtype(data.dtype, np.integer):
        return data.astype(np.float64)
    return data


def compute_log_sum_exp(data, axes, keepdims):
    data = _cast_integer_to_float(data)
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

        data = _cast_integer_to_float(data)
        if data.size == 0:
            return self.reduce_constant(data, -np.inf, tax, keepdims)
        return compute_log_sum_exp(data, tax, keepdims)


class ReduceLogSumExp_18(OpRunReduceNumpy):
    def _run(self, data, axes=None, keepdims=1, noop_with_empty_axes=0):
        axes = self.handle_axes(axes, noop_with_empty_axes)

        keepdims = keepdims != 0

        data = _cast_integer_to_float(data)
        if data.size == 0:
            return self.reduce_constant(data, -np.inf, axes, keepdims)

        return compute_log_sum_exp(data, axes, keepdims)
