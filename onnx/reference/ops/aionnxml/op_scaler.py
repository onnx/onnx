# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from onnx.reference.ops.aionnxml._op_run_aionnxml import OpRunAiOnnxMl

# pylint: disable=R0913,R0914,W0221


class Scaler(OpRunAiOnnxMl):
    def _run(self, x, offset=None, scale=None):  # type: ignore
        dx = x - offset
        return ((dx * scale).astype(x.dtype),)
