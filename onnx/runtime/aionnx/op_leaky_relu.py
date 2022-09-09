# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from ._op import OpRunUnaryNum


def _leaky_relu(x: numpy.ndarray, alpha: float) -> numpy.ndarray:
    sign = (x > 0).astype(x.dtype)
    sign -= ((sign - 1) * alpha).astype(x.dtype)
    return x * sign


class LeakyRelu(OpRunUnaryNum):
    def _run(self, x):  # type: ignore
        return (_leaky_relu(x, self.alpha),)  # type: ignore
