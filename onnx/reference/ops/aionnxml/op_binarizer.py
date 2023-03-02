# SPDX-License-Identifier: Apache-2.0
# pylint: disable=R0913,R0914,W0221

import numpy as np

from onnx.reference.ops.aionnxml._op_run_aionnxml import OpRunAiOnnxMl


def compute_binarizer(x, threshold=None):
    y = x.copy()
    cond = y > threshold
    not_cond = np.logical_not(cond)
    y[cond] = 1
    y[not_cond] = 0
    return (y,)


class Binarizer(OpRunAiOnnxMl):
    def _run(self, x, threshold=None):  # type: ignore
        return compute_binarizer(x, threshold)
