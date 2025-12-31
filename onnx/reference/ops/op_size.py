# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.op_run import OpRun


class Size(OpRun):
    def _run(self, data):
        xp = self._get_array_api_namespace(data)
        # Return array using the same namespace as input
        size_val = data.size
        # Create array with int64 dtype using array API
        result = xp.asarray([size_val], dtype=xp.int64)
        # Return as 0-d array (scalar)
        return (result[0],)
