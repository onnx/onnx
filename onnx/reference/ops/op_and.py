# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from onnx.reference.ops._op import OpRunBinary


class And(OpRunBinary):
    def _run(self, x, y):
        xp = self._get_array_api_namespace(
            *[
                a
                for a in [x, y]
                if hasattr(a, "__array_namespace__") or hasattr(a, "shape")
            ]
        )
        xp = self._get_array_api_namespace(x, y)
        return (xp.logical_and(x, y),)
