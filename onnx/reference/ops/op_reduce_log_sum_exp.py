# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy as np

from ._op import OpRunReduceNumpy


class ReduceLogSumExp(OpRunReduceNumpy):
    def _run(self, data, axes=None, keepdims=None):  # type: ignore
        tax = tuple(axes) if axes else None
        data_max = data.copy()
        ind = np.isinf(data_max)
        data_max[ind] = -np.inf
        mx = data_max.max(axis=tax, keepdims=True)
        sub = np.subtract(data, mx)
        exp = np.exp(sub, out=sub)
        mxs = np.sum(exp, axis=tax, keepdims=True, dtype=data.dtype)
        res = np.log(mxs) + mx
        if not keepdims:  # type: ignore
            res = np.squeeze(res, axis=tax)
        return (res,)
