# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations


from onnx.reference.op_run import OpRun


def _one_hot(indices, depth, axis=-1, dtype=np.float32):
    values = np.asarray(indices)
    rank = len(values.shape)
    depth_range = np.arange(depth)
    if axis < 0:
        axis += rank + 1
    ls = values.shape[0:axis]
    rs = values.shape[axis:rank]
    new_shape = (1,) * len(ls) + depth_range.shape + (1,) * len(rs)
    targets = xp.reshape(depth_range, new_shape)
    values = xp.reshape(np.mod(values, depth), (*ls, 1, *rs))
    return np.asarray(targets == values, dtype=dtype)


class OneHot(OpRun):
    def _run(self, indices, depth, values, axis=None):
        xp = self._get_array_api_namespace(indices)
        off_value, on_value = values
        y = _one_hot(indices, depth, axis=axis, dtype=values.dtype)
        y = y * (on_value - off_value) + off_value
        return (y,)