# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy as np  # type: ignore

from ..op_run import OpRun


class Pow(OpRun):
    def _run(self, a, b):  # type: ignore
        return (np.power(a, b).astype(a.dtype),)
