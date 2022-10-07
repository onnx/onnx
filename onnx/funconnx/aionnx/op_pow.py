# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

from warnings import catch_warnings, simplefilter

import numpy as np  # type: ignore

from ..op_run import OpRun


class Pow(OpRun):
    def _run(self, a, b):  # type: ignore
        with catch_warnings():
            simplefilter("ignore")
            return (np.power(a, b).astype(a.dtype),)
