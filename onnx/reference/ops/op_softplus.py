# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from onnx.reference.ops._op import OpRunUnaryNum


class Softplus(OpRunUnaryNum):
    def _run(self, X):
        xp = self._get_array_api_namespace(X)
        # Softplus(x) = log(exp(x) + 1)
        return (xp.log(xp.exp(X) + 1),)
