# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.op_run import OpRun


class CumProd(OpRun):
    def _run(self, x, axis, exclusive=None, reverse=None):
        axis = np.asarray(axis)
        if axis.ndim != 0:
            raise ValueError(f"Axis must be a rank-0 tensor, got `{axis.ndim}`.")
        if reverse:
            rev_indices = [slice(0, s) for s in x.shape]
            rev_indices[axis] = slice(None, None, -1)
            x = x[tuple(rev_indices)]
        if exclusive:
            indices_c = [slice(0, s) for s in x.shape]
            indices_d = [slice(0, s) for s in x.shape]
            indices_c[axis] = slice(0, -1)
            indices_d[axis] = slice(1, x.shape[axis])
            res = np.ones(x.shape, dtype=x.dtype)
            np.cumprod(x[tuple(indices_c)], axis=axis, out=res[tuple(indices_d)])
        else:
            res = np.cumprod(x, axis=axis, dtype=x.dtype)
        if reverse:
            res = res[tuple(rev_indices)]
        return (res,)