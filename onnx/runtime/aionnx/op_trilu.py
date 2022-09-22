# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy as np  # type: ignore

from ..op_run import OpRun


class Trilu(OpRun):
    def _run(self, x, k=None):  # type: ignore
        k = 0 if k is None else int(k)
        if self.upper:  # type: ignore
            return (np.triu(x, k),)
        return (np.tril(x, k),)
