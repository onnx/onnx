# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from onnx.reference.ops._op import OpRunUnaryNum


class LeakyRelu(OpRunUnaryNum):
    def _run(self, x, alpha=None):
        xp = self._get_array_api_namespace(x)
        alpha = alpha or self.alpha
        # LeakyRelu(x) = x if x > 0 else alpha * x
        zero = xp.asarray(0, dtype=x.dtype)
        return (xp.where(x > zero, x, alpha * x),)