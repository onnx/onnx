# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from ._op import OpRunUnaryNum


class Hardmax(OpRunUnaryNum):
    def _run(self, x, axis=None):  # type: ignore
        axis = axis or self.axis  # type: ignore
        x_argmax = numpy.argmax(x, axis=axis)  # type: ignore
        y = numpy.zeros_like(x)
        numpy.put_along_axis(
            y, numpy.expand_dims(x_argmax, axis=axis), 1, axis=axis  # type: ignore
        )
        return (y,)
