# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.op_run import OpRun


def _one_hot(indices, depth, axis=-1, dtype=np.float32):
    values = np.asarray(indices)
    depth = np.asarray(depth)
    if depth.ndim > 1 or depth.size != 1:
        raise ValueError(
            "OneHot: 'depth' must be a scalar or a rank-1 tensor with exactly "
            f"one element, got shape {depth.shape}."
        )
    # A non-integer depth is cast to int64 before use.
    depth = int(depth.astype(np.int64).item())
    if depth < 0:
        raise ValueError(f"OneHot: 'depth' must be non-negative, got {depth}.")
    # Indices are cast to int64 before use. This also widens narrow integer
    # dtypes such as int8, so that wrapping negative indices with depth cannot
    # overflow. uint64 is kept: it is never negative, and casting values above
    # the int64 range would turn them into wrapped, in-range negative indices.
    if values.dtype != np.uint64:
        values = values.astype(np.int64, copy=False)
        # Wrap negative in-range indices; leave out-of-range ones unmatched (all off_value).
        values = np.where(values < 0, values + depth, values)
    rank = len(values.shape)
    depth_range = np.arange(depth)
    if axis < 0:
        axis += rank + 1
    ls = values.shape[0:axis]
    rs = values.shape[axis:rank]
    new_shape = (1,) * len(ls) + depth_range.shape + (1,) * len(rs)
    targets = np.reshape(depth_range, new_shape)
    values = np.reshape(values, (*ls, 1, *rs))
    return np.asarray(targets == values, dtype=dtype)


class OneHot(OpRun):
    def _run(self, indices, depth, values, axis=None):
        off_value, on_value = values
        mask = _one_hot(indices, depth, axis=axis, dtype=np.bool_)
        y = np.where(mask, on_value, off_value).astype(values.dtype)
        return (y,)
