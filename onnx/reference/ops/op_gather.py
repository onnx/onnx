# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.array_api_namespace import asarray, convert_to_numpy
from onnx.reference.op_run import OpRun


class Gather(OpRun):
    def _run(self, x, indices, axis=None):
        xp = self._get_array_api_namespace(x, indices)
        # Array API doesn't have take, use numpy for gather
        x_np = convert_to_numpy(x)
        indices_np = convert_to_numpy(indices)

        if not x_np.flags["C_CONTIGUOUS"]:
            x_np = np.ascontiguousarray(x_np)
        if not indices_np.flags["C_CONTIGUOUS"]:
            indices_np = np.ascontiguousarray(indices_np)
        if indices_np.size == 0:
            result = np.empty((0,), dtype=x_np.dtype)
        else:
            try:
                result = np.take(x_np, indices_np, axis=axis)
            except TypeError:
                # distribution x86 requires int32.
                result = np.take(x_np, indices_np.astype(int), axis=axis)

        # Convert back to original array type
        return (asarray(result, xp=xp),)
