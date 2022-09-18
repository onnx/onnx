# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import math

import numpy  # type: ignore

from ..op_run import OpRun


class LRN(OpRun):
    def _run(self, x):  # type: ignore
        # TODO: support overridden attributes.
        if len(x.shape) != 4:
            raise RuntimeError(
                f"LRN only applies on 4D tensors but shape is {x.shape!r}."
            )
        square_sum = numpy.zeros(x.shape).astype(x.dtype)
        for ind in numpy.ndindex(x.shape):
            n, c, h, w = ind
            begin = max(0, c - int(math.floor((self.size - 1) / 2)))  # type: ignore
            end = min(5, c + int(math.ceil((self.size - 1) / 2)) + 1)  # type: ignore
            square_sum[n, c, h, w] = numpy.sum(x[n, begin:end, h, w] ** 2)
        y = x / ((self.bias + (self.alpha / self.size) * square_sum) ** self.beta)  # type: ignore
        return (y.astype(x.dtype),)
