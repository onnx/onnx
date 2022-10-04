# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy as np  # type: ignore

from ._op import OpRunReduceNumpy


class ReduceLogSum(OpRunReduceNumpy):
    def _run(self, data, axes=None, keepdims=None):  # type: ignore
        tax = tuple(axes) if axes else None
        res = np.sum(data, axis=tax, keepdims=keepdims)
        if len(res.shape) > 0:
            return (np.log(res, out=res),)
        return (np.log(res),)
