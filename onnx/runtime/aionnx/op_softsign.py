# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from ._op import OpRunUnaryNum


class Softsign(OpRunUnaryNum):
    def _run(self, X):  # type: ignore
        tmp = numpy.abs(X)
        tmp += 1
        numpy.divide(X, tmp, out=tmp)
        return (tmp,)
