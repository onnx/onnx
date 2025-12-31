# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from onnx.reference.ops._op import OpRunUnaryNum


class Relu(OpRunUnaryNum):
    def _run(self, x):
        xp = self._get_array_api_namespace(x)
        # Array API doesn't have a relu function, but we can use maximum
        zero = xp.asarray(0, dtype=x.dtype)
        return (xp.maximum(x, zero),)
