# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from onnx.reference.ops._op import OpRunUnaryNum


class Sigmoid(OpRunUnaryNum):
    def _run(self, X):
        xp = self._get_array_api_namespace(X)
        if len(X.shape) == 0 or X.size == 0:
            # For scalars and empty arrays, use manual sigmoid
            # Array API doesn't have a sigmoid function
            return (1 / (1 + xp.exp(-X)),)
        # Use the standard sigmoid formula: 1 / (1 + exp(-x))
        return (1 / (1 + xp.exp(-X)),)