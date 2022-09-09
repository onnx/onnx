# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from ..op_run import OpRun


class Range(OpRun):
    def _run(self, starts, ends, steps):  # type: ignore
        return (numpy.arange(starts, ends, steps).astype(starts.dtype),)
