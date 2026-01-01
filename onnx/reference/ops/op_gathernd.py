# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

from onnx.reference.op_run import OpRun


def _gather_nd_impl(data: Any, indices: Any, batch_dims: int) -> tuple[Any]:
    # Note the data rank - will be reused multiple times later
    data_rank = len(data.shape)

    # The list of data/indice shape of batch_dims.
    batch_dims_shape = []

    # The number of elements in the batch_dims for data/indice array.
    batch_dims_size = 1

    # Check the shape of indice and data are identical for batch dims.
    for i in range(batch_dims):
        batch_dims_shape.append(indices.shape[i])
        batch_dims_size *= indices.shape[i]

    # Compute output of the op as below.
    # Compute shape of output array.
    output_shape = (
        batch_dims_shape + list(indices.shape)[batch_dims:-1]
        if (indices.shape[-1] == data_rank - batch_dims)
        else batch_dims_shape
        + list(indices.shape)[batch_dims:-1]
        + list(data.shape)[batch_dims + indices.shape[-1] :]
    )

    # Placeholder for output data.

    # Flatten 'indices' to 2D array.
    reshaped_indices = indices.reshape(batch_dims_size, -1, indices.shape[-1])

    # Flatten 'data' to array of shape
    # (batch_dim_size, data.shape[batch_dimes:]).
    reshaped_data = data.reshape((batch_dims_size, *data.shape[batch_dims:]))

    # Gather each scalar value from 'data'.
    output_list = []
    for batch_dim in range(reshaped_indices.shape[0]):
        for outer_dim in range(reshaped_indices.shape[1]):
            gather_index = tuple(reshaped_indices[batch_dim][outer_dim])
            output_list.append(reshaped_data[(batch_dim, *gather_index)])

    # Use array API to construct result
    from onnx.reference.array_api_namespace import asarray, get_array_api_namespace

    xp = get_array_api_namespace(data)

    # Convert list to array - use concat if available, otherwise stack
    if len(output_list) > 0:
        # Create array from list of scalars
        result = xp.stack([asarray(x, xp=xp) for x in output_list])
        result = xp.reshape(result, output_shape)
    else:
        result = xp.zeros(output_shape, dtype=data.dtype)
    return (result,)


class GatherND(OpRun):
    def _run(self, data, indices, batch_dims=None):
        self._get_array_api_namespace(data)
        return _gather_nd_impl(data, indices, batch_dims)
