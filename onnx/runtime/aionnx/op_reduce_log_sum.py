# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from ._op import OpRunReduceNumpy


class ReduceLogSum(OpRunReduceNumpy):
    def _run(self, data):  # type: ignore
        # TODO: support overridden attributes.
        tax = tuple(self.axes) if self.axes else None  # type: ignore
        res = numpy.sum(data, axis=tax, keepdims=self.keepdims)  # type: ignore
        if len(res.shape) > 0:
            return (numpy.log(res, out=res),)
        return (numpy.log(res),)
