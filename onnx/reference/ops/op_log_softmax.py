# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy as np

from .op_softmax import Softmax


class LogSoftmax(Softmax):
    def _run(self, X):  # type: ignore
        Y = Softmax._run(self, X)[0]
        np.log(Y, out=Y)
        return (Y,)
