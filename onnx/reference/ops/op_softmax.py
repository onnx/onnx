# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from onnx.reference.ops._op import OpRunUnaryNum


class Softmax(OpRunUnaryNum):
    def _run(self, X, axis=None):
        xp = self._get_array_api_namespace(X)

        if X.size == 0:
            return (X,)

        axis = axis or self.axis

        # Use array API functions
        tmp = X - xp.max(X, axis=axis, keepdims=True)
        Y = xp.exp(tmp)
        Y = Y / xp.sum(Y, axis=axis, keepdims=True)

        # Ensure dtype is preserved
        if hasattr(Y, "dtype") and Y.dtype != X.dtype:
            if hasattr(xp, "astype"):
                Y = xp.astype(Y, X.dtype)
            else:
                Y = Y.astype(X.dtype)

        return (Y,)
