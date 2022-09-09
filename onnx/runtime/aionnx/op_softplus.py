# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from ._op import OpRunUnaryNum


class Softplus(OpRunUnaryNum):
    def _run(self, X):  # type: ignore
        tmp = numpy.exp(X)
        tmp += 1
        numpy.log(tmp, out=tmp)
        return (tmp,)
