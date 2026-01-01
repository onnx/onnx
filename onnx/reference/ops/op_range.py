# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any


from onnx.reference.op_run import OpRun


class Range(OpRun):
    def _run(self, starts, ends, steps):
        xp = self._get_array_api_namespace(starts)
        start_scalar = starts.item()
        if isinstance(ends, Any):
            ends = ends.item()
        if isinstance(steps, Any):
            steps = steps.item()
        return (np.arange(start_scalar, ends, steps).astype(starts.dtype),)