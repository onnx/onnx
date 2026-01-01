# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from onnx.reference.op_run import OpRun


class Det(OpRun):
    def _run(self, x):
        xp = self._get_array_api_namespace(x)
        # Use array API linalg.det
        det_val = xp.linalg.det(x)
        # Return as array
        return (det_val,)
