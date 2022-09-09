# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from ..op_run import OpRun


class NonZero(OpRun):
    def _run(self, x):  # type: ignore
        res = numpy.vstack(numpy.nonzero(x))
        return (res,)
