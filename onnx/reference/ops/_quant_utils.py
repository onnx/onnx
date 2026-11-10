# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np


def reshape_input(
    value: np.ndarray,
    shape: tuple[int, ...],
    axis: int,
    block_size: int | None = None,
) -> np.ndarray:
    """Reshape/Replicate scale/zero-point to be broadcastable to shape.

    Args:
        value: the array to be reshaped/replicated
        shape: the target shape
        axis: quantization axis, applicable for per-axis and blocked quantization
        block_size: size of quantization block, applicable only for blocked quantization

    Returns:
        value array after reshape/replicate according to quantization mode.
    """
    if len(value.shape) == 0:
        return value
    if len(value.shape) > 0 and value.size == 1:
        return value[0]
    if not block_size:
        assert len(value.shape) == 1
        dims = [1] * len(shape)
        try:
            dims[axis] = value.size
            return value.reshape(tuple(dims))
        except IndexError as e:
            raise IndexError(
                f"axis is out of boundary, axis={axis}, "
                f"value.shape={value.shape}, shape={shape}."
            ) from e

    if block_size <= 0:
        raise ValueError("block_size must be a positive integer.")

    # repeat scale to get element-wise scale
    value = np.repeat(value, repeats=block_size, axis=axis)
    if (
        shape[axis] != value.shape[axis]
    ):  # block_size does not divide x, handle the remainder block
        value = value.take(indices=range(shape[axis]), axis=axis)
    if value.shape != shape:
        raise ValueError(
            "Invalid shapes for Blocked Quantization. Input 2 shape should identical to Input 1 shape, except for one dimension, in which blocking is performed"
        )
    assert np.broadcast_shapes(shape, value.shape) == shape
    return value


def reshape_multi_axis(
    value: np.ndarray, shape: tuple[int, ...], block_shape: list[int]
) -> np.ndarray:
    if len(block_shape) != len(shape):
        raise ValueError("block_shape must have one entry per input dimension.")
    if any(
        isinstance(size, (bool, np.bool_))
        or not isinstance(size, (int, np.integer))
        or size <= 0
        for size in block_shape
    ):
        raise ValueError("block_shape entries must be positive integers.")
    grid_shape = tuple(
        (dim + size - 1) // size for dim, size in zip(shape, block_shape, strict=True)
    )
    if value.shape != grid_shape:
        raise ValueError(
            f"Expected scale/zero-point grid shape {grid_shape}, got {value.shape}."
        )
    for axis, size in enumerate(block_shape):
        value = np.repeat(value, min(size, shape[axis]), axis=axis)
        value = value[tuple(slice(dim) for dim in shape)]
    return value[tuple(slice(dim) for dim in shape)]
