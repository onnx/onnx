# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from onnx.reference.ops._op import OpRunBinaryComparison


class GreaterOrEqual(OpRunBinaryComparison):
    def _run(self, a, b):
        xp = self._get_array_api_namespace(a, b)
        return (xp.greater_equal(a, b),)
