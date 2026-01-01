# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from onnx.reference.op_run import OpRun


class Selu(OpRun):
    def _run(self, x, alpha=None, gamma=None):
        xp = self._get_array_api_namespace(x)
        return (xp.where(x > 0, x, xp.exp(x) * alpha - alpha) * gamma,)