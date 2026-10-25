# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.op_run import OpRun


def _one_hot(indices, depth, axis=-1, dtype=np.float32):
    values = np.asarray(indices)
    # Non-integer indices and depth are cast to int64 before use, and depth
    # may be a scalar or a rank-1 tensor holding exactly one element.
    if not np.issubdtype(values.dtype, np.integer):
        values = values.astype(np.int64)
    depth = int(np.asarray(depth).astype(np.int64).reshape(-1)[0])
    rank = len(values.shape)
    depth_range = np.arange(depth)
    if axis < 0:
        axis += rank + 1
    ls = values.shape[0:axis]
    rs = values.shape[axis:rank]
    new_shape = (1,) * len(ls) + depth_range.shape + (1,) * len(rs)
    targets = np.reshape(depth_range, new_shape)
    # Wrap negative in-range indices; leave out-of-range ones unmatched (all off_value).
    values = np.where(values < 0, values + depth, values)
    values = np.reshape(values, (*ls, 1, *rs))
    return np.asarray(targets == values, dtype=dtype)


class OneHot(OpRun):
    def _run(self, indices, depth, values, axis=None):
        off_value, on_value = values
        mask = _one_hot(indices, depth, axis=axis, dtype=np.bool_)
        y = np.where(mask, on_value, off_value).astype(values.dtype)
        return (y,)
