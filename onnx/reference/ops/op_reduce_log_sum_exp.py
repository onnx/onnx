# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.ops._op import OpRunReduceNumpy


def compute_log_sum_exp(data, axes, keepdims, xp):
    data_max = data.copy() if hasattr(data, "copy") else data + xp.zeros_like(data)
    ind = xp.isinf(data_max) if hasattr(xp, "isinf") else np.isinf(np.asarray(data_max))
    if hasattr(ind, "__array_namespace__"):
        data_max = xp.where(
            ind, xp.asarray(-xp.inf if hasattr(xp, "inf") else -np.inf), data_max
        )
    else:
        data_max[ind] = -np.inf
    mx = xp.max(data_max, axis=axes, keepdims=True)
    sub = data - mx
    exp = xp.exp(sub)
    mxs = xp.sum(exp, axis=axes, keepdims=True)
    res = xp.log(mxs) + mx
    if not keepdims:
        res = np.squeeze(np.asarray(res), axis=axes)
        from onnx.reference.array_api_namespace import asarray

        res = asarray(xp, res)
    return (res,)


class ReduceLogSumExp_1(OpRunReduceNumpy):
    def _run(self, data, axes=None, keepdims=None):
        xp = self._get_array_api_namespace(data)
        tax = tuple(axes) if axes is not None else None

        if data.size == 0:
            return self.reduce_constant(data, -np.inf, tax, keepdims)
        return compute_log_sum_exp(data, tax, keepdims, xp)


class ReduceLogSumExp_18(OpRunReduceNumpy):
    def _run(self, data, axes=None, keepdims=1, noop_with_empty_axes=0):
        xp = self._get_array_api_namespace(data)
        axes = self.handle_axes(axes, noop_with_empty_axes)

        keepdims = keepdims != 0

        if data.size == 0:
            return self.reduce_constant(data, -np.inf, axes, keepdims)

        return compute_log_sum_exp(data, axes, keepdims, xp)
