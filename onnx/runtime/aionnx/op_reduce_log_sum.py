# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy as np  # type: ignore

from ._op import OpRunReduceNumpy


class ReduceLogSum(OpRunReduceNumpy):
    def _run(self, data):  # type: ignore
        # TODO: support overridden attributes.
        tax = tuple(self.axes) if self.axes else None  # type: ignore
        res = np.sum(data, axis=tax, keepdims=self.keepdims)  # type: ignore
        if len(res.shape) > 0:
            return (np.log(res, out=res),)
        return (np.log(res),)
