# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from onnx.reference.op_run import OpRun


def _scatter_nd_impl(data, indices, updates, reduction=None, xp=None):
    # Create a copy using array API
    from onnx.reference.array_api_namespace import get_array_api_namespace
    if xp is None:
        xp = get_array_api_namespace(data)
    
    # Create output as a copy of data
    # Use addition with zeros to create a copy
    output = data + xp.zeros_like(data)
    
    # Iterate through indices - handle multi-dimensional indices properly
    # indices.shape is (..., K) where K is the index dimension
    indices_shape = indices.shape
    if len(indices_shape) == 1:
        # Single index case
        idx = tuple(int(x) for x in indices)
        if reduction == "add":
            output[idx] = output[idx] + updates
        elif reduction == "mul":
            output[idx] = output[idx] * updates
        elif reduction == "max":
            output[idx] = xp.maximum(output[idx], updates)
        elif reduction == "min":
            output[idx] = xp.minimum(output[idx], updates)
        else:
            output[idx] = updates
    else:
        # Multiple indices case - iterate over first dimensions
        num_indices = 1
        for dim in indices_shape[:-1]:
            num_indices *= dim
        
        # Flatten the indices to iterate
        flat_indices = xp.reshape(indices, (num_indices, indices_shape[-1]))
        flat_updates = xp.reshape(updates, (num_indices, *updates.shape[len(indices_shape)-1:]))
        
        for i in range(num_indices):
            idx = tuple(int(flat_indices[i, j]) for j in range(indices_shape[-1]))
            if reduction == "add":
                output[idx] = output[idx] + flat_updates[i]
            elif reduction == "mul":
                output[idx] = output[idx] * flat_updates[i]
            elif reduction == "max":
                output[idx] = xp.maximum(output[idx], flat_updates[i])
            elif reduction == "min":
                output[idx] = xp.minimum(output[idx], flat_updates[i])
            else:
                output[idx] = flat_updates[i]
    return output


class ScatterND(OpRun):
    def _run(self, data, indices, updates, reduction=None):
        xp = self._get_array_api_namespace(data)
        y = _scatter_nd_impl(data, indices, updates, reduction=reduction, xp=xp)
        return (y,)
