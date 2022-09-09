# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from ._op import OpRunUnary


class Not(OpRunUnary):
    def _run(self, x):  # type: ignore
        return (numpy.logical_not(x),)
