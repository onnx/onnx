# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from ..op_run import OpRun


class OptionalGetElement(OpRun):
    def _run(self, x):  # type: ignore
        if x is None:
            return ([],)
        if isinstance(x, list):
            if len(x) > 0:
                return (x[0],)
        elif isinstance(x, numpy.ndarray):
            if len(x.shape) > 0 and x.shape[0] > 0:
                return (x[0],)
        raise RuntimeError("Input is empty.")
