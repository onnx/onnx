# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from ._op import OpRunUnaryNum


class HardSigmoid(OpRunUnaryNum):
    def _run(self, x):  # type: ignore
        y = numpy.maximum(0, numpy.minimum(1, x * self.alpha + self.beta))  # type: ignore
        return (y,)
