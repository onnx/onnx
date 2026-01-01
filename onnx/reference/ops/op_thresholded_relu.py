# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from onnx.reference.ops._op import OpRunUnaryNum


class ThresholdedRelu(OpRunUnaryNum):
    def _run(self, x, alpha=None):
        xp = self._get_array_api_namespace(x)
        alpha = alpha or self.alpha
        zero = xp.asarray(0, dtype=x.dtype)
        return (xp.where(x > alpha, x, zero),)