# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from ._op import OpRunUnaryNum


class ThresholdedRelu(OpRunUnaryNum):
    def _run(self, x, alpha=None):  # type: ignore
        alpha = alpha or self.alpha  # type: ignore
        return (numpy.where(x > alpha, x, 0),)  # type: ignore
