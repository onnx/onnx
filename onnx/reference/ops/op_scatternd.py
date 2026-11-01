# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.op_run import OpRun
from onnx.reference.ops._scatter import scatter_reduce


def _scatter_nd_impl(data, indices, updates, reduction=None, include_self=1):
    if reduction == "mean" or include_self != 1:
        update_pairs = (
            (tuple(indices[i]), updates[i]) for i in np.ndindex(indices.shape[:-1])
        )
        return scatter_reduce(data, update_pairs, reduction or "none", include_self)
    output = np.copy(data)
    for i in np.ndindex(indices.shape[:-1]):
        if reduction == "add":
            output[tuple(indices[i])] += updates[i]
        elif reduction == "mul":
            output[tuple(indices[i])] *= updates[i]
        elif reduction == "max":
            output[tuple(indices[i])] = np.maximum(
                output[tuple(indices[i])], updates[i]
            )
        elif reduction == "min":
            output[tuple(indices[i])] = np.minimum(
                output[tuple(indices[i])], updates[i]
            )
        else:
            output[tuple(indices[i])] = updates[i]
    return output


class ScatterND(OpRun):
    def _run(self, data, indices, updates, reduction=None, include_self=1):
        y = _scatter_nd_impl(
            data, indices, updates, reduction=reduction, include_self=include_self
        )
        return (y,)
