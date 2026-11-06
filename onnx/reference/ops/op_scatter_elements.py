# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.op_run import OpRun


def scatter_elements(data, indices, updates, axis=0, reduction=None):
    """Scatter elements.

    ::
        for 3-dim and axis=0
            output[indices[i][j][k]][j][k] = updates[i][j][k]
        for axis 1
            output[i][indices[i][j][k]][k] = updates[i][j][k]
        and so on.
    """
    if reduction == "add":

        def f(x, y):
            return x + y

    elif reduction == "mul":

        def f(x, y):
            return x * y

    elif reduction == "min":

        def f(x, y):
            return min(x, y)

    elif reduction == "max":

        def f(x, y):
            return max(x, y)

    else:

        def f(x, y):  # noqa: ARG001
            return y

    if axis < 0:
        axis = data.ndim + axis

    scattered = np.copy(data)
    for update_index in np.ndindex(indices.shape):
        output_index = list(update_index)
        output_index[axis] = indices[update_index]
        output_index = tuple(output_index)
        scattered[output_index] = f(scattered[output_index], updates[update_index])
    return scattered


class ScatterElements(OpRun):
    def _run(self, data, indices, updates, axis=None, reduction=None):
        res = scatter_elements(data, indices, updates, axis=axis, reduction=reduction)
        return (res,)
