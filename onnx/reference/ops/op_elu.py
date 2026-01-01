# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from onnx.reference.ops._op import OpRunUnaryNum


class Elu(OpRunUnaryNum):
    def _run(self, x, alpha=None):
        xp = self._get_array_api_namespace(x)
        alpha = alpha or self.alpha
        return (xp.where(x > 0, x, alpha * (xp.exp(x) - 1)),)