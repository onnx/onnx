# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy as np  # type: ignore

from ._op import OpRunUnaryNum


class Shrink(OpRunUnaryNum):
    def _run(self, x):  # type: ignore
        # TODO: support overridden attributes.
        return (
            np.where(
                x < -self.lambd,  # type: ignore
                x + self.bias,  # type: ignore
                np.where(x > self.lambd, x - self.bias, 0),  # type: ignore
            ),
        )
