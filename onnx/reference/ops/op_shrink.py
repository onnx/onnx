# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from onnx.reference.op_run import OpRun


class Shrink(OpRun):
    def _run(self, x, bias=None, lambd=None):
        xp = self._get_array_api_namespace(x)
        return (
            xp.where(
                x < -lambd,
                x + bias,
                xp.where(x > lambd, x - bias, 0),
            ).astype(x.dtype),
        )
