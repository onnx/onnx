# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from onnx.reference.ops._op import OpRunUnaryNum


class Softsign(OpRunUnaryNum):
    def _run(self, X):
        xp = self._get_array_api_namespace(X)
        # Softsign(x) = x / (1 + |x|)
        return (X / (1 + xp.abs(X)),)