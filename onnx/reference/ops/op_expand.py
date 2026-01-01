# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from onnx.reference.op_run import OpRun


class Expand(OpRun):
    def _run(self, data, shape):
        xp = self._get_array_api_namespace(data)
        # Use broadcast_to for expansion
        ones = xp.ones(tuple(shape), dtype=data.dtype)
        return (data * ones,)