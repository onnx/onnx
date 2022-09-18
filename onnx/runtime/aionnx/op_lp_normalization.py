# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from ._op import OpRunUnaryNum


class LpNormalization(OpRunUnaryNum):
    def _run(self, x, axis=None, p=None):  # type: ignore
        axis = axis or self.axis  # type: ignore
        p = p or self.p  # type: ignore
        norm = numpy.power(numpy.power(x, p).sum(axis=axis), 1.0 / p)  # type: ignore
        norm = numpy.expand_dims(norm, axis)  # type: ignore
        return (x / norm,)
