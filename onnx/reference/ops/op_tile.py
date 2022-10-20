# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy as np

from ..op_run import OpRun


class Tile(OpRun):
    def _run(self, x, repeats):  # type: ignore
        return (np.tile(x, repeats),)
