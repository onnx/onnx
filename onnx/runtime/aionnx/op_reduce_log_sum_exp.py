# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from ._op import OpRunReduceNumpy


class ReduceLogSumExp(OpRunReduceNumpy):
    def _run(self, data):  # type: ignore
        # TODO: support overridden attributes.
        tax = tuple(self.axes) if self.axes else None  # type: ignore
        data_max = data.copy()
        ind = numpy.isinf(data_max)
        data_max[ind] = -numpy.inf
        mx = data_max.max(axis=tax, keepdims=True)
        sub = numpy.subtract(data, mx)
        exp = numpy.exp(sub, out=sub)
        mxs = numpy.sum(exp, axis=tax, keepdims=True, dtype=data.dtype)
        res = numpy.log(mxs) + mx
        if not self.keepdims:  # type: ignore
            res = numpy.squeeze(res, axis=tax)
        return (res,)
