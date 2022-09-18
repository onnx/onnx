# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from ._op import OpRunUnaryNum


class HardSigmoid(OpRunUnaryNum):
    def _run(self, x, alpha=None, beta=None):  # type: ignore
        alpha = alpha or self.alpha  # type: ignore
        beta = beta or self.beta  # type: ignore
        y = numpy.maximum(0, numpy.minimum(1, x * alpha + beta))
        return (y,)
