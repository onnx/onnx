# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ml_dtypes
import numpy as np

from onnx.reference.ops._op import OpRunReduceNumpy


def _reduce_sum(x, axes, keepdims):
    acc_dtype = (
        np.float32
        if x.dtype in (np.float16, np.dtype(ml_dtypes.bfloat16))
        else x.dtype
    )
    res = np.sum(
        x.astype(acc_dtype, copy=False),
        axis=axes,
        keepdims=keepdims,
        dtype=acc_dtype,
    ).astype(x.dtype, copy=False)
    if keepdims == 0 and not isinstance(res, np.ndarray):
        # The runtime must return a numpy array of a single float.
        res = np.array(res)
    return res


class ReduceSum_1(OpRunReduceNumpy):
    def _run(self, x, axes=None, keepdims=None):
        axes = tuple(axes) if axes is not None else None
        return (_reduce_sum(x, axes, keepdims),)


class ReduceSum_13(OpRunReduceNumpy):
    def _run(self, x, axes=None, keepdims=None, noop_with_empty_axes=None):
        axes = self.handle_axes(axes, noop_with_empty_axes)

        try:
            res = _reduce_sum(x, axes, keepdims)
        except TypeError as e:
            raise TypeError(
                f"Unable to reduce shape {x.shape!r} with axes={axes!r} and keepdims={keepdims}."
            ) from e
        else:
            return (res,)
