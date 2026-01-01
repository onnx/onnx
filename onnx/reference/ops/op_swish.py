# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from onnx.reference.ops._op import OpRunUnaryNum


class Swish(OpRunUnaryNum):
    def _run(self, x, alpha=None):
        xp = self._get_array_api_namespace(x)
        alpha = self.alpha if alpha is None else alpha
        # Swish(x) = x * sigmoid(alpha * x)
        return (x * (1 / (1 + xp.exp(-alpha * x))),)