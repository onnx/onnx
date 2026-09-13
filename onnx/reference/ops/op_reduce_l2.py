# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.ops._op import OpRunReduceNumpy


def _reduce_l2(data, axes, keepdims):
    if not np.issubdtype(data.dtype, np.floating):
        return np.sqrt(np.sum(np.square(data), axis=axes, keepdims=keepdims)).astype(
            dtype=data.dtype
        )

    # Scaling first avoids overflowing or underflowing the intermediate
    # squares when the final norm is representable.
    working_dtype = np.float64 if data.dtype == np.float64 else np.float32
    working = data.astype(working_dtype, copy=False)
    with np.errstate(invalid="ignore", over="ignore", under="ignore"):
        scale = np.max(np.abs(working), axis=axes, keepdims=True, initial=0)
        finite_nonzero = np.isfinite(scale) & (scale != 0)
        scaled = working / np.where(finite_nonzero, scale, 1)
        stable = scale * np.sqrt(np.sum(np.square(scaled), axis=axes, keepdims=True))

        # Preserve the previous Inf/NaN behavior instead of defining new
        # non-finite semantics as part of a finite-range correction.
        direct = np.sqrt(np.sum(np.square(working), axis=axes, keepdims=True))
        result = np.where(np.isfinite(scale), stable, direct)

    if not keepdims:
        if axes is None:
            result = np.squeeze(result)
        elif axes != ():
            result = np.squeeze(result, axis=axes)
    return result.astype(dtype=data.dtype)


class ReduceL2_1(OpRunReduceNumpy):
    def _run(self, data, axes=None, keepdims=None):
        axes = tuple(axes) if axes is not None else None
        res = _reduce_l2(data, axes, keepdims)
        if keepdims == 0 and not isinstance(res, np.ndarray):
            # The runtime must return a numpy array of a single float.
            res = np.array(res)
        return (res,)


class ReduceL2_18(OpRunReduceNumpy):
    def _run(self, data, axes=None, keepdims=1, noop_with_empty_axes=0):
        axes = self.handle_axes(axes, noop_with_empty_axes)

        keepdims = keepdims != 0
        res = _reduce_l2(data, axes, keepdims)
        if keepdims == 0 and not isinstance(res, np.ndarray):
            # The runtime must return a numpy array of a single float.
            res = np.array(res)
        return (res,)
