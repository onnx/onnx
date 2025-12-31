# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from onnx.reference.ops._op import OpRunBinary


class BitwiseOr(OpRunBinary):
    def _run(self, x, y):
        xp = self._get_array_api_namespace(x, y)
        return (xp.bitwise_or(x, y),)
