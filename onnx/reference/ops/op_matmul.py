# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from onnx.reference.ops._op import OpRunBinaryNum


class MatMul(OpRunBinaryNum):
    def _run(self, a, b):
        xp = self._get_array_api_namespace(a, b)
        return (xp.matmul(a, b),)
