# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from onnx.reference.ops._op import OpRunUnary


class BitwiseNot(OpRunUnary):
    def _run(self, X):
        xp = self._get_array_api_namespace(X)
        return (xp.bitwise_invert(X),)
