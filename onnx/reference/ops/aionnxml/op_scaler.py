# SPDX-License-Identifier: Apache-2.0
# pylint: disable=R0913,R0914,W0221

from ._op_run_aionnxml import OpRunAiOnnxMl


class Scaler(OpRunAiOnnxMl):

    op_domain = "ai.onnx.ml"

    def _run(self, x, offset=None, scale=None):
        dx = x - offset
        return (dx * scale,)
