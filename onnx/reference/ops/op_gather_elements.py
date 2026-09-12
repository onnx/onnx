# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.op_run import OpRun


def gather_numpy(self: np.ndarray, dim: int, index: np.ndarray) -> np.ndarray:
    dim %= self.ndim
    data_slices = tuple(
        slice(None) if axis == dim else slice(0, size)
        for axis, size in enumerate(index.shape)
    )
    return np.take_along_axis(self[data_slices], index, axis=dim)


class GatherElements(OpRun):
    def _run(self, data, indices, axis=None):
        if indices.size == 0:
            return (np.empty(indices.shape, dtype=data.dtype),)
        try:
            return (gather_numpy(data, axis, indices),)
        except TypeError:
            # distribution x86 requires int32.
            return (gather_numpy(data, axis, indices.astype(int)),)
