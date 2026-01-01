# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

from onnx.reference.op_run import OpRun


def _vcelu1(x: Any, alpha: float = 1.0) -> Any:
    positive_input = xp.maximum(0, x)
    negative_input = xp.minimum(0, alpha * (xp.exp(x / alpha) - 1))
    return positive_input + negative_input


class Celu(OpRun):
    def _run(self, x, alpha=None):
        self._get_array_api_namespace(x)
        return (_vcelu1(x, alpha).astype(x.dtype),)
