# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.op_run import OpRun


class CumSum(OpRun):
    def _run(self, x, axis, exclusive=None, reverse=None):
        if not isinstance(axis, (np.int32, np.int64)):
            if len(axis.shape) > 1 or (len(axis.shape) > 0 and axis.shape[0] != 1):
                raise RuntimeError(
                    f"axis must be an array of one number not {axis} (shape {axis.shape})."
                )
            if len(axis.shape) > 0:
                axis = axis[0]
        if reverse:
            rev_indices = [slice(0, s) for s in x.shape]
            rev_indices[axis] = slice(None, None, -1)
            x = x[tuple(rev_indices)]
        if exclusive:
            indices_c = [slice(0, s) for s in x.shape]
            indices_d = [slice(0, s) for s in x.shape]
            indices_c[axis] = slice(0, -1)
            indices_d[axis] = slice(1, x.shape[axis])
            res = np.zeros(x.shape, dtype=x.dtype)
            np.cumsum(x[tuple(indices_c)], axis=axis, out=res[tuple(indices_d)])
        else:
            res = np.cumsum(x, axis=axis, dtype=x.dtype)
        if reverse:
            res = res[tuple(rev_indices)]
        return (res,)
