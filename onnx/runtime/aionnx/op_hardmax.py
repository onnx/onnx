# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from ._op import OpRunUnaryNum


class Hardmax(OpRunUnaryNum):
    def _run(self, x):  # type: ignore
        x_argmax = numpy.argmax(x, axis=self.axis)  # type: ignore
        y = numpy.zeros_like(x)
        numpy.put_along_axis(
            y, numpy.expand_dims(x_argmax, axis=self.axis), 1, axis=self.axis  # type: ignore
        )
        return (y,)
