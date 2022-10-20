# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy as np

from ._op import OpRunUnaryNum


class Softplus(OpRunUnaryNum):
    def _run(self, X):  # type: ignore
        tmp = np.exp(X)
        tmp += 1
        np.log(tmp, out=tmp)
        return (tmp,)
