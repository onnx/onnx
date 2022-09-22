# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy as np  # type: ignore

from ._op import OpRunUnaryNum


class Selu(OpRunUnaryNum):
    def _run(self, x, alpha=None, gamma=None):  # type: ignore
        alpha = alpha or self.alpha  # type: ignore
        gamma = gamma or self.gamma  # type: ignore
        return (np.where(x > 0, x, np.exp(x) * alpha - alpha) * gamma,)
