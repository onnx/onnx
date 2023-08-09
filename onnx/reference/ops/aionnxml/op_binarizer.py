# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from onnx.reference.ops.aionnxml._op_run_aionnxml import OpRunAiOnnxMl

# pylint: disable=R0913,R0914,W0221


def compute_binarizer(x, threshold=None):
    return ((x > threshold).astype(x.dtype),)


class Binarizer(OpRunAiOnnxMl):
    def _run(self, x, threshold=None):  # type: ignore
        return compute_binarizer(x, threshold)
