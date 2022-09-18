# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from ..op_run import OpRun


class Mod(OpRun):
    def _run(self, a, b, fmod=None):  # type: ignore
        fmod = fmod or self.fmod  # type: ignore
        if fmod == 1:  # type: ignore
            return (numpy.fmod(a, b),)
        if a.dtype in (numpy.float16, numpy.float32, numpy.float64):
            return (numpy.nan_to_num(numpy.fmod(a, b)),)
        return (numpy.nan_to_num(numpy.mod(a, b)),)
