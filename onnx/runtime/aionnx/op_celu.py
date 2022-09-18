# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from ._op import OpRunUnaryNum


def _vcelu1(x: numpy.ndarray, alpha: float = 1.0) -> numpy.ndarray:
    positive_input = numpy.maximum(0, x)
    negative_input = numpy.minimum(0, alpha * (numpy.exp(x / alpha) - 1))
    return positive_input + negative_input


class Celu(OpRunUnaryNum):
    def _run(self, x):  # type: ignore
        # TODO: support overridden attributes.
        return (_vcelu1(x, self.alpha),)  # type: ignore
