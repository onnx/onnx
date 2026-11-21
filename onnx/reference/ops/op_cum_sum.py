# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ml_dtypes
import numpy as np

from onnx.reference.op_run import OpRun


class CumSum(OpRun):
    def _run(self, x, axis, exclusive=None, reverse=None):
        axis = np.asarray(axis)
        if axis.ndim != 0:
            raise ValueError(f"Axis must be a rank-0 tensor, got `{axis.ndim}`.")
        acc_dtype = (
            np.float32
            if x.dtype in (np.float16, np.dtype(ml_dtypes.bfloat16))
            else x.dtype
        )
        xc = x.astype(acc_dtype, copy=False)
        if reverse:
            rev_indices = [slice(0, s) for s in x.shape]
            rev_indices[axis] = slice(None, None, -1)
            xc = xc[tuple(rev_indices)]
        if exclusive:
            indices_c = [slice(0, s) for s in x.shape]
            indices_d = [slice(0, s) for s in x.shape]
            indices_c[axis] = slice(0, -1)
            indices_d[axis] = slice(1, x.shape[axis])
            res = np.zeros(x.shape, dtype=acc_dtype)
            np.cumsum(
                xc[tuple(indices_c)],
                axis=axis,
                dtype=acc_dtype,
                out=res[tuple(indices_d)],
            )
        else:
            res = np.cumsum(xc, axis=axis, dtype=acc_dtype)
        if reverse:
            res = res[tuple(rev_indices)]
        return (res.astype(x.dtype, copy=False),)
