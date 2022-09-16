# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from ..op_run import OpRun


class MatMulInteger(OpRun):
    def _run(self, A, B, a_zero_point=None, b_zero_point=None):  # type: ignore
        A32 = A.astype(numpy.int32)
        if a_zero_point is not None:
            A32 -= a_zero_point
        B32 = B.astype(numpy.int32)
        if b_zero_point is not None:
            B32 -= b_zero_point
        return (A32 @ B32,)
