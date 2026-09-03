# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math

import numpy as np

from onnx.reference.op_run import OpRun


class MaxUnpool(OpRun):
    def _run(
        self, X, indices, output_shape=None, kernel_shape=None, pads=None, strides=None
    ):
        batch_and_channel_dims = 2
        if X.ndim < batch_and_channel_dims:
            raise ValueError(
                "Input X must have at least "
                f"{batch_and_channel_dims} dimensions, but has shape {X.shape}."
            )
        if indices.shape != X.shape:
            raise ValueError(
                f"Indices shape {indices.shape} must match input shape {X.shape}."
            )
        if indices.dtype != np.int64:
            raise TypeError(
                f"Indices must have dtype int64, but has dtype {indices.dtype}."
            )

        pooling_dims = X.ndim - batch_and_channel_dims
        kernel_shape = self.kernel_shape if kernel_shape is None else kernel_shape
        pads = self.pads if pads is None else pads
        strides = self.strides if strides is None else strides

        if kernel_shape is None:
            raise ValueError("Attribute kernel_shape must be specified.")
        if len(kernel_shape) != pooling_dims:
            raise ValueError(
                f"kernel_shape must have {pooling_dims} values, but has {len(kernel_shape)}."
            )
        if any(kernel <= 0 for kernel in kernel_shape):
            raise ValueError("kernel_shape values must be positive.")
        if strides is None:
            strides = [1] * pooling_dims
        elif len(strides) != pooling_dims:
            raise ValueError(
                f"strides must have {pooling_dims} values, but has {len(strides)}."
            )
        if any(stride <= 0 for stride in strides):
            raise ValueError("strides values must be positive.")
        if pads is None:
            pads = [0] * (pooling_dims * 2)
        elif len(pads) != pooling_dims * 2:
            raise ValueError(
                f"pads must have {pooling_dims * 2} values, but has {len(pads)}."
            )
        if any(pad < 0 for pad in pads):
            raise ValueError("pads values must be non-negative.")

        if output_shape is not None:
            if output_shape.ndim != 1:
                raise ValueError("output_shape must be a rank-1 tensor.")
            if output_shape.dtype != np.int64:
                raise TypeError(
                    f"output_shape must have dtype int64, but has dtype {output_shape.dtype}."
                )
            if output_shape.size != X.ndim:
                raise ValueError(
                    "output_shape must have the same number of elements as the rank of X."
                )
            shape = tuple(int(dim) for dim in output_shape)
            if any(dim < 0 for dim in shape):
                raise ValueError(
                    f"output_shape must have non-negative dimensions, but is {shape}."
                )
        else:
            shape = X.shape[:batch_and_channel_dims] + tuple(
                (X.shape[dim + batch_and_channel_dims] - 1) * strides[dim]
                - pads[dim]
                - pads[pooling_dims + dim]
                + kernel_shape[dim]
                for dim in range(pooling_dims)
            )
            if any(dim < 0 for dim in shape):
                raise ValueError(
                    f"Inferred output shape must be non-negative, but is {shape}."
                )

        output_size = math.prod(shape)
        flat_indices = indices.reshape(-1)
        if np.any(flat_indices < 0) or np.any(flat_indices >= output_size):
            raise ValueError(
                f"Indices must be in [0, {output_size}), but received values outside that range."
            )

        result = np.zeros(shape, dtype=X.dtype)
        last_positions = np.full(output_size, -1, dtype=np.intp)
        np.maximum.at(last_positions, flat_indices, np.arange(X.size))
        written = last_positions >= 0
        result.reshape(-1)[written] = X.reshape(-1)[last_positions[written]]
        return (result,)
