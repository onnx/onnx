# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy as np  # type: ignore

from ..op_run import OpRun


def _vcelu1(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    positive_input = np.maximum(0, x)
    negative_input = np.minimum(0, alpha * (np.exp(x / alpha) - 1))
    return positive_input + negative_input  # type: ignore


class Celu(OpRun):
    def _run(self, x, alpha=None):  # type: ignore
        return (_vcelu1(x, alpha),)
