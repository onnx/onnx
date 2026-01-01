# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.op_run import OpRun


class Range(OpRun):
    def _run(self, starts, ends, steps):
        xp = self._get_array_api_namespace(starts)
        start_scalar = starts.item()
        if isinstance(ends, np.ndarray):
            ends = ends.item()
        if isinstance(steps, np.ndarray):
            steps = steps.item()
        return (np.arange(start_scalar, ends, steps).astype(starts.dtype),)