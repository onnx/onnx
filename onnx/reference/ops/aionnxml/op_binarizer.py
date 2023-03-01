# SPDX-License-Identifier: Apache-2.0
# pylint: disable=R0913,R0914,W0221

import numpy as np

from onnx.reference.ops.aionnxml._op_run_aionnxml import OpRunAiOnnxMl


class Binarizer(OpRunAiOnnxMl):
    def _run(self, x, threshold=None):  # type: ignore
        X = x.copy()
        cond = threshold < X
        not_cond = np.logical_not(cond)
        X[cond] = 1
        X[not_cond] = 0
        return (X,)
