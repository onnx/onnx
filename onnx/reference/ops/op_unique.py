# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.op_run import OpRun


def _specify_int64(indices, inverse_indices, counts):
    return (
        np.array(indices, dtype=np.int64),
        np.array(inverse_indices, dtype=np.int64),
        np.array(counts, dtype=np.int64),
    )


class Unique(OpRun):
    def _run(self, x, axis=None, sorted=None):  # type: ignore[override]  # noqa: A002
        if axis is not None and np.isnan(axis):
            axis = None
        if axis is None:
            y, indices, inverse_indices, counts = np.unique(x, True, True, True)
        else:
            y, indices, inverse_indices, counts = np.unique(
                x, True, True, True, axis=axis
            )
        # numpy 2.0 returns inverse_indices with the shape of x when axis is None,
        # numpy 1.x and ONNX use a flat tensor.
        inverse_indices = np.reshape(inverse_indices, (-1,))

        if not sorted:
            # np.unique always sorts, so put the unique values, their indices and
            # their counts back into order of first occurrence.
            argsorted_indices = np.argsort(indices)
            inverse_indices_map = dict(
                zip(argsorted_indices, np.arange(len(argsorted_indices)), strict=True)
            )
            y = np.take(y, argsorted_indices, axis=0 if axis is None else axis)
            indices = indices[argsorted_indices]
            inverse_indices = np.asarray(
                [inverse_indices_map[i] for i in inverse_indices], dtype=np.int64
            )
            counts = counts[argsorted_indices]

        if len(self.onnx_node.output) == 1:
            return (y,)

        indices, inverse_indices, counts = _specify_int64(
            indices, inverse_indices, counts
        )
        if len(self.onnx_node.output) == 2:
            return (y, indices)
        if len(self.onnx_node.output) == 3:
            return (y, indices, inverse_indices)
        return (y, indices, inverse_indices, counts)
