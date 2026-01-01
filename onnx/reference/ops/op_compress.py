# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.array_api_namespace import asarray, convert_to_numpy
from onnx.reference.op_run import OpRun


class Compress(OpRun):
    def _run(self, x, condition, axis=None):
        xp = self._get_array_api_namespace(x, condition)
        # compress not in array API, use numpy
        x_np = convert_to_numpy(x)
        condition_np = convert_to_numpy(condition)
        result = np.compress(condition_np, x_np, axis=axis)
        return (asarray(result, xp=xp),)
