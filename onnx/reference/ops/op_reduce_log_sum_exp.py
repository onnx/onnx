# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy as np

from onnx.defs import onnx_opset_version

from ._op import OpRunReduceNumpy


def compute_log_sum_exp(data, axes, keepdims):
    data_max = data.copy()
    ind = np.isinf(data_max)
    data_max[ind] = -np.inf
    mx = data_max.max(axis=axes, keepdims=True)
    sub = np.subtract(data, mx)
    exp = np.exp(sub, out=sub)
    mxs = np.sum(exp, axis=axes, keepdims=True, dtype=data.dtype)
    res = np.log(mxs) + mx
    if not keepdims:  # type: ignore
        res = np.squeeze(res, axis=axes)
    return (res,)

class ReduceLogSumExp_1_11_13(OpRunReduceNumpy):
    def run(self, data, axes=None, keepdims=None):  # type: ignore
        tax = tuple(axes) if axes else None
        return compute_log_sum_exp(data, axes, keepdims)


class ReduceLogSumExp_18(OpRunReduceNumpy):
    def run(self, data, axes=None):  # type: ignore
        return self._run(data, axes)

    def _run(self, data, axes):  # type: ignore
        if self.IsAxesEmpty(axes) and self.noop_with_empty_axes:
            return (data,)

        axes = self.HandleAxes(axes)
        keepdims = self.keepdims != 0

        return compute_log_sum_exp(data, axes, keepdims)


if onnx_opset_version() >= 18:
    ReduceLogSumExp = ReduceLogSumExp_18
else:
    ReduceLogSumExp = ReduceLogSumExp_1_11_13  # type: ignore
