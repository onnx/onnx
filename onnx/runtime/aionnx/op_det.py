# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from ..op_run import OpRun


class Det(OpRun):
    def _run(self, x):  # type: ignore
        res = numpy.linalg.det(x)
        if not isinstance(res, numpy.ndarray):
            res = numpy.array([res])
        return (res,)
