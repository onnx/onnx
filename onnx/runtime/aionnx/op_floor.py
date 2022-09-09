# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from ._op import OpRunUnaryNum


class Floor(OpRunUnaryNum):
    def _run(self, x):  # type: ignore
        return (numpy.floor(x),)
