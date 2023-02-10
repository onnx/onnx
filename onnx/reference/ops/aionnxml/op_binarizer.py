# SPDX-License-Identifier: Apache-2.0
# pylint: disable=R0913,R0914,W0221

from ....backend.test.case.node.ai_onnx_ml.binarizer import compute_binarizer
from ._op_run_aionnxml import OpRunAiOnnxMl


class Binarizer(OpRunAiOnnxMl):
    def _run(self, x, threshold=None):  # type: ignore
        return compute_binarizer(x, threshold)
