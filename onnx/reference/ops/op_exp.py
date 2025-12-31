# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from onnx.reference.ops._op import OpRunUnaryNum


class Exp(OpRunUnaryNum):
    def _run(self, x):
        xp = self._get_array_api_namespace(x)
        result = xp.exp(x)
        # Ensure dtype is preserved
        if hasattr(result.dtype, 'type') and result.dtype != x.dtype:
            if hasattr(xp, 'astype'):
                result = xp.astype(result, x.dtype)
            else:
                result = result.astype(x.dtype)
        return (result,)
