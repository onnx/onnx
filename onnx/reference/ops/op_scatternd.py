# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.array_api_namespace import asarray, convert_to_numpy, get_array_api_namespace
from onnx.reference.op_run import OpRun


def _scatter_nd_impl(data, indices, updates, reduction=None):
    # Get array namespace
    xp = get_array_api_namespace(data)
    
    # Convert to numpy for operations not in array API
    data_np = convert_to_numpy(data)
    indices_np = convert_to_numpy(indices)
    updates_np = convert_to_numpy(updates)
    
    output = np.copy(data_np)
    for i in np.ndindex(indices_np.shape[:-1]):
        if reduction == "add":
            output[tuple(indices_np[i])] += updates_np[i]
        elif reduction == "mul":
            output[tuple(indices_np[i])] *= updates_np[i]
        elif reduction == "max":
            output[tuple(indices_np[i])] = np.maximum(output[tuple(indices_np[i])], updates_np[i])
        elif reduction == "min":
            output[tuple(indices_np[i])] = np.minimum(output[tuple(indices_np[i])], updates_np[i])
        else:
            output[tuple(indices_np[i])] = updates_np[i]
    
    # Convert back to original array type
    return asarray(output, xp=xp)


class ScatterND(OpRun):
    def _run(self, data, indices, updates, reduction=None):
        self._get_array_api_namespace(data)
        y = _scatter_nd_impl(data, indices, updates, reduction=reduction)
        return (y,)
