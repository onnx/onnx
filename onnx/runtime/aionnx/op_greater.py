# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from ._op import OpRunBinaryComparison


class Greater(OpRunBinaryComparison):
    def _run(self, a, b):  # type: ignore
        return (numpy.greater(a, b),)
