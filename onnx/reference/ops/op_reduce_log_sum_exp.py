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


def _check_integer_input(data, op_name):
    # ReduceLogSumExp and ReduceLogSum are composite operators built on Log/Exp,
    # which do not support integer types. Opsets <= 18 list integers in their
    # type constraints by mistake (see onnx/onnx#7141); opset 21 removes them.
    if np.issubdtype(data.dtype, np.integer):
        raise TypeError(
            f"{op_name} does not support integer types (got {data.dtype}). "
            f"Log/Exp are undefined for integers; cast the input to a float type. "
            f"Integer support was removed from opset 21."
        )


class ReduceLogSumExp_1(OpRunReduceNumpy):
    def _run(self, data, axes=None, keepdims=None):
        tax = tuple(axes) if axes is not None else None

        if data.size == 0:
            return self.reduce_constant(data, -np.inf, tax, keepdims)
        _check_integer_input(data, "ReduceLogSumExp")
        return compute_log_sum_exp(data, tax, keepdims)


class ReduceLogSumExp_18(OpRunReduceNumpy):
    def _run(self, data, axes=None, keepdims=1, noop_with_empty_axes=0):
        _check_integer_input(data, "ReduceLogSumExp")
        axes = self.handle_axes(axes, noop_with_empty_axes)

        keepdims = keepdims != 0

        if data.size == 0:
            return self.reduce_constant(data, -np.inf, axes, keepdims)

        return compute_log_sum_exp(data, axes, keepdims)


class ReduceLogSumExp_21(OpRunReduceNumpy):
    def _run(self, data, axes=None, keepdims=1, noop_with_empty_axes=0):
        _check_integer_input(data, "ReduceLogSumExp")
        axes = self.handle_axes(axes, noop_with_empty_axes)

        keepdims = keepdims != 0

        if data.size == 0:
            return self.reduce_constant(data, -np.inf, axes, keepdims)

        return compute_log_sum_exp(data, axes, keepdims)
