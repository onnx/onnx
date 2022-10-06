# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy as np  # type: ignore

from ..op_run import OpRun


class PRelu(OpRun):
    def _run(self, x, slope):  # type: ignore
        print(x)
        print(slope)
        return (np.where(x > 0, x, x * slope),)
