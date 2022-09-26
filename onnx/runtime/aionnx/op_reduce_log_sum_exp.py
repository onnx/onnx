# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy as np  # type: ignore

from ._op import OpRunReduceNumpy


class ReduceLogSumExp(OpRunReduceNumpy):
    def _run(self, data):  # type: ignore
        # TODO: support overridden attributes.
        tax = tuple(self.axes) if self.axes else None  # type: ignore
        data_max = data.copy()
        ind = np.isinf(data_max)
        data_max[ind] = -np.inf
        mx = data_max.max(axis=tax, keepdims=True)
        sub = np.subtract(data, mx)
        exp = np.exp(sub, out=sub)
        mxs = np.sum(exp, axis=tax, keepdims=True, dtype=data.dtype)  # type: ignore
        res = np.log(mxs) + mx
        if not self.keepdims:  # type: ignore
            res = np.squeeze(res, axis=tax)
        return (res,)
