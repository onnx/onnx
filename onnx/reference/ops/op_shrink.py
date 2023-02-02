# SPDX-License-Identifier: Apache-2.0
# pylint: disable=E1130,W0221

import numpy as np

from onnx.reference.op_run import OpRun


class Shrink(OpRun):
    def _run(self, x, bias=None, lambd=None):  # type: ignore
        return (
            np.where(
                x < -lambd,
                x + bias,
                np.where(x > lambd, x - bias, 0),
            ).astype(x.dtype),
        )
