# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.ops._op import OpRunReduceNumpy


def _mean(data, axes, keepdims):
    if data.size == 0 or not np.issubdtype(data.dtype, np.inexact):
        return np.mean(data, axis=axes, keepdims=keepdims, dtype=data.dtype)

    # Scaling keeps the intermediate sum in range without changing the mean.
    # Use one as the scale for zero and non-finite slices so their prior NumPy
    # semantics are preserved.
    scale = np.max(np.abs(data), axis=axes, keepdims=True)
    safe_scale = np.where(np.isfinite(scale) & (scale != 0), scale, 1)
    result = (
        np.mean(
            data / safe_scale,
            axis=axes,
            keepdims=True,
            dtype=data.dtype,
        )
        * safe_scale
    )
    return result if keepdims else np.squeeze(result, axis=axes)


class ReduceMean_1(OpRunReduceNumpy):
    def _run(self, data, axes=None, keepdims=None):
        axes = tuple(axes) if axes is not None else None
        res = _mean(data, axes=axes, keepdims=keepdims)
        if keepdims == 0 and not isinstance(res, np.ndarray):
            # The runtime must return a numpy array of a single float.
            res = np.array(res)
        return (res,)


class ReduceMean_18(OpRunReduceNumpy):
    def _run(self, data, axes=None, keepdims=1, noop_with_empty_axes=0):
        axes = self.handle_axes(axes, noop_with_empty_axes)

        keepdims = keepdims != 0
        try:
            res = _mean(data, axes=axes, keepdims=keepdims)
            if keepdims == 0 and not isinstance(res, np.ndarray):
                # The runtime must return a numpy array of a single float.
                res = np.array(res)
        except TypeError as e:
            raise TypeError(
                f"Unable to reduce shape {data.shape!r} with axes={axes!r} and keepdims={keepdims}."
            ) from e
        return (res,)
