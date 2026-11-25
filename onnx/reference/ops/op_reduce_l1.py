# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ml_dtypes
import numpy as np

from onnx.reference.ops._op import OpRunReduceNumpy


def _reduce_l1(data: np.ndarray, axes: tuple[int, ...] | None, keepdims: int | bool):
    # Accumulate float16 and bfloat16 in float32: bfloat16 stalls at 256 when
    # summing ones.
    working = (
        data.astype(np.float32)
        if data.dtype in (np.float16, ml_dtypes.bfloat16)
        else data
    )
    res = np.sum(np.abs(working), axis=axes, keepdims=keepdims).astype(dtype=data.dtype)
    if keepdims == 0 and not isinstance(res, np.ndarray):
        # The runtime must return a numpy array of a single float.
        res = np.array(res)
    return res


class ReduceL1_1(OpRunReduceNumpy):
    def _run(self, data, axes=None, keepdims=None):
        axes = tuple(axes) if axes is not None else None
        return (_reduce_l1(data, axes, keepdims),)


class ReduceL1_18(OpRunReduceNumpy):
    def _run(self, data, axes=None, keepdims=1, noop_with_empty_axes=0):
        axes = self.handle_axes(axes, noop_with_empty_axes)

        keepdims = keepdims != 0
        return (_reduce_l1(data, axes, keepdims),)
