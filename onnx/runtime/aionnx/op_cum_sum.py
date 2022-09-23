# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy as np  # type: ignore

from ..op_run import OpRun


class CumSum(OpRun):
    def _run(self, x, *axis):  # type: ignore
        axis = None if len(axis) == 0 else axis[0]  # type: ignore
        if axis is None:
            if self.reverse or self.exclusive:
                raise NotImplementedError("reverse=1 or exclusive=1 not implemented")
            return (np.cumsum(x),)
        if not isinstance(axis, (np.int32, np.int64)):
            if len(axis.shape) > 1 or (len(axis.shape) > 0 and axis.shape[0] != 1):  # type: ignore
                raise RuntimeError(
                    f"axis must be an array of one number not {axis} (shape {axis.shape})."  # type: ignore
                )
            if len(axis.shape) > 0:  # type: ignore
                axis = axis[0]
        if self.reverse:  # type: ignore
            rev_indices = [slice(0, s) for s in x.shape]
            rev_indices[axis] = slice(None, None, -1)  # type: ignore
            x = x[tuple(rev_indices)]
        if self.exclusive:  # type: ignore
            indices_c = [slice(0, s) for s in x.shape]
            indices_d = [slice(0, s) for s in x.shape]
            indices_c[axis] = slice(0, -1)  # type: ignore
            indices_d[axis] = slice(1, x.shape[axis])  # type: ignore
            res = np.zeros(x.shape, dtype=x.dtype)
            np.cumsum(x[tuple(indices_c)], axis=axis, out=res[tuple(indices_d)])  # type: ignore
        else:
            res = np.cumsum(x, axis=axis)  # type: ignore
        if self.reverse:  # type: ignore
            res = res[tuple(rev_indices)]
        return (res,)
