# SPDX-License-Identifier: Apache-2.0
# pylint: disable=R0913,W0221

import numpy as np

from onnx.reference.op_run import OpRun


def scatter_elements(data, indices, updates, axis=0, reduction=None):  # type: ignore
    """
    ::
        // for 3-dim and axis=0
        //    output[indices[i][j][k]][j][k] = updates[i][j][k]
        // for axis 1
        //    output[i][indices[i][j][k]][k] = updates[i][j][k]
        // and so on
    """
    if reduction == "add":
        f = lambda x, y: x + y  # noqa
    elif reduction == "min":
        f = lambda x, y: min(x, y)  # noqa
    elif reduction == "max":
        f = lambda x, y: max(x, y)  # noqa
    else:
        f = lambda x, y: y  # noqa

    if axis < 0:
        axis = data.ndim + axis

    if len(data.shape) == 1 and axis == 0:
        scattered = np.copy(data)
        for pos, up in zip(indices, updates):
            scattered[pos] = f(scattered[pos], up)
        return scattered

    if len(indices.shape) == 2:
        scattered = np.copy(data)
        if axis == 0:
            for i in range(indices.shape[0]):
                for j in range(indices.shape[1]):
                    scattered[indices[i, j], j] = f(
                        scattered[indices[i, j], j], updates[i, j]
                    )
        else:
            for i in range(indices.shape[0]):
                for j in range(indices.shape[1]):
                    scattered[i, indices[i, j]] = f(
                        scattered[i, indices[i, j]], updates[i, j]
                    )
        return scattered

    if len(indices.shape) == 3:
        scattered = np.copy(data)
        if axis == 0:
            for i in range(indices.shape[0]):
                for j in range(indices.shape[1]):
                    for k in range(indices.shape[2]):
                        scattered[indices[i, j, k], j, k] = f(
                            scattered[indices[i, j, k], j, k], updates[i, j, k]
                        )
        elif axis == 1:
            for i in range(indices.shape[0]):
                for j in range(indices.shape[1]):
                    for k in range(indices.shape[2]):
                        scattered[i, indices[i, j, k], k] = f(
                            scattered[i, indices[i, j, k], k], updates[i, j, k]
                        )
        elif axis == 2:
            for i in range(indices.shape[0]):
                for j in range(indices.shape[1]):
                    for k in range(indices.shape[2]):
                        scattered[i, j, indices[i, j, k]] = f(
                            scattered[i, j, indices[i, j, k]], updates[i, j, k]
                        )
        return scattered

    idx_xsection_shape = indices.shape[:axis] + indices.shape[axis + 1 :]

    def make_slice(arr, axis, i):  # type: ignore
        slc = [slice(None)] * arr.ndim
        slc[axis] = i
        return slc

    def unpack(packed):  # type: ignore
        unpacked = packed[0]
        for i in range(1, len(packed)):
            unpacked = unpacked, packed[i]
        return unpacked

    # We use indices and axis parameters to create idx
    # idx is in a form that can be used as a NumPy advanced
    # indices for scattering of updates param. in data
    idx = [
        [
            unpack(np.indices(idx_xsection_shape).reshape(indices.ndim - 1, -1)),
            indices[tuple(make_slice(indices, axis, i))].reshape(1, -1)[0],
        ]
        for i in range(indices.shape[axis])
    ]
    idx = list(np.concatenate(idx, axis=1))
    idx.insert(axis, idx.pop())

    # updates_idx is a NumPy advanced indices for indexing
    # of elements in the updates
    updates_idx = list(idx)
    updates_idx.pop(axis)
    updates_idx.insert(  # type: ignore
        axis,
        np.repeat(np.arange(indices.shape[axis]), np.prod(idx_xsection_shape)),  # type: ignore
    )

    scattered = np.copy(data)
    if reduction == "min":
        print()
        print("I", idx, updates_idx)
        print(scattered[tuple(idx)], updates[tuple(updates_idx)])
        print("*", np.minimum(scattered[tuple(idx)], updates[tuple(updates_idx)]))
        scattered[tuple(idx)] = np.minimum(
            scattered[tuple(idx)], updates[tuple(updates_idx)]
        )
        print("F", scattered)
    elif reduction == "max":
        scattered[tuple(idx)] = np.maximum(
            scattered[tuple(idx)], updates[tuple(updates_idx)]
        )
    elif reduction == "add":
        scattered[tuple(idx)] += updates[tuple(updates_idx)]
    else:
        scattered[tuple(idx)] = updates[tuple(updates_idx)]
    return scattered


class ScatterElements(OpRun):
    def _run(self, data, indices, updates, axis=None, reduction=None):  # type: ignore
        res = scatter_elements(data, indices, updates, axis=axis, reduction=reduction)
        return (res,)
