# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from onnx.reference.ops._op import OpRunUnaryNum


class Neg(OpRunUnaryNum):
    def _run(self, x):
        xp = self._get_array_api_namespace(x)
        return (xp.negative(x),)
