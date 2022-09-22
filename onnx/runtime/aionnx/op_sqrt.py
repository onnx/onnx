# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy as np  # type: ignore

from ._op import OpRunUnaryNum


class Sqrt(OpRunUnaryNum):
    def _run(self, x):  # type: ignore
        return (np.sqrt(x),)
