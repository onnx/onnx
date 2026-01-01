# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from onnx.reference.ops._op import OpRunUnaryNum


class Atanh(OpRunUnaryNum):
    def _run(self, x):
        xp = self._get_array_api_namespace(
            *[
                a
                for a in [x]
                if hasattr(a, "__array_namespace__") or hasattr(a, "shape")
            ]
        )
        xp = self._get_array_api_namespace(x)
        return (xp.atanh(x),)
