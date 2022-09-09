# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from ._op import OpRunUnaryNum


class Reciprocal(OpRunUnaryNum):
    def _run(self, x):  # type: ignore
        with numpy.errstate(divide="ignore"):
            return (numpy.reciprocal(x),)
