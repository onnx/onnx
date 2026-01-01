# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from onnx.reference.ops._op import OpRunUnaryNum


class HardSigmoid(OpRunUnaryNum):
    def _run(self, x, alpha=None, beta=None):
        xp = self._get_array_api_namespace(x)
        alpha = alpha or self.alpha
        beta = beta or self.beta
        y = xp.clip(x * alpha + beta, 0, 1)
        return (y,)
