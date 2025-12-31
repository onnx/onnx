# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from onnx.reference.ops._op import OpRunUnary


class IsNaN(OpRunUnary):
    def _run(self, data):
        xp = self._get_array_api_namespace(data)
        return (xp.isnan(data),)
