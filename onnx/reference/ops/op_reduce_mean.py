# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math

import numpy as np

from onnx.reference.ops._op import OpRunReduceNumpy


def _mean(data, axes, keepdims):
    # Preserve NumPy's precision and fast path for ordinary finite results.
    with np.errstate(over="ignore", invalid="ignore"):
        direct = np.mean(data, axis=axes, keepdims=keepdims, dtype=data.dtype)
    if data.size == 0 or not np.issubdtype(data.dtype, np.floating):
        return direct

    if np.all(np.isfinite(direct)):
        return direct

    if axes is None:
        reduction_axes = tuple(range(data.ndim))
    else:
        reduction_axes = tuple(axis % data.ndim for axis in axes)

    if not reduction_axes or len(set(reduction_axes)) != len(reduction_axes):
        return direct

    remaining_axes = tuple(
        axis for axis in range(data.ndim) if axis not in reduction_axes
    )
    remaining_shape = tuple(data.shape[axis] for axis in remaining_axes)
    reduction_size = math.prod(data.shape[axis] for axis in reduction_axes)
    rows = np.transpose(data, remaining_axes + reduction_axes).reshape(
        -1, reduction_size
    )
    result = np.asarray(direct).reshape(-1).copy()

    for index in np.flatnonzero(~np.isfinite(result)):
        row = rows[index]
        if not np.all(np.isfinite(row)):
            # A non-finite input is not an intermediate-overflow failure.
            continue

        try:
            total = math.fsum(float(value) for value in row)
            value = total / reduction_size
        except OverflowError:
            # Dividing each finite term first bounds the positive and negative
            # partial totals by the largest representable input magnitude.
            value = math.fsum(float(item) / reduction_size for item in row)
        result[index] = value

    reduced = result.reshape(remaining_shape)
    if keepdims:
        output_shape = tuple(
            1 if axis in reduction_axes else data.shape[axis]
            for axis in range(data.ndim)
        )
        return reduced.reshape(output_shape)
    return reduced


class ReduceMean_1(OpRunReduceNumpy):
    def _run(self, data, axes=None, keepdims=None):
        axes = tuple(axes) if axes is not None else None
        res = _mean(data, axes=axes, keepdims=keepdims)
        if keepdims == 0 and not isinstance(res, np.ndarray):
            # The runtime must return a numpy array of a single float.
            res = np.array(res)
        return (res,)


class ReduceMean_18(OpRunReduceNumpy):
    def _run(self, data, axes=None, keepdims=1, noop_with_empty_axes=0):
        axes = self.handle_axes(axes, noop_with_empty_axes)

        keepdims = keepdims != 0
        try:
            res = _mean(data, axes=axes, keepdims=keepdims)
            if keepdims == 0 and not isinstance(res, np.ndarray):
                # The runtime must return a numpy array of a single float.
                res = np.array(res)
        except TypeError as e:
            raise TypeError(
                f"Unable to reduce shape {data.shape!r} with axes={axes!r} and keepdims={keepdims}."
            ) from e
        return (res,)
