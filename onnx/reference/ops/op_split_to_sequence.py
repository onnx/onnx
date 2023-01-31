# SPDX-License-Identifier: Apache-2.0
# pylint: disable=C0200,W0221

import numpy as np

from onnx.reference.op_run import OpRun


class SplitToSequence(OpRun):
    def common_run(self, mat, split, axis, length):  # type: ignore
        if split is None:
            split = [1 for _ in range(mat.shape[axis])]
        elif len(split.shape) == 0:
            # A scaler
            dim = mat.shape[axis]
            length = int(split)
            n = dim // int(length)
            split = [length] * n
            left = dim - length * n
            if left > 0:
                split.append(left)

        sli = [slice(0, s) for s in mat.shape]
        res = []
        pos = 0
        for spl in split:
            sli[axis] = slice(pos, pos + spl)  # type: ignore
            pos += spl
            res.append(mat[tuple(sli)])
        return tuple(res)

    def _run(self, mat, split=None, axis=None, keepdims=None):  # type: ignore
        if split is None:
            length = None
        else:
            if len(split.shape) == 0:
                length = split
            else:
                length = split.shape[0]
        res = list(self.common_run(mat, split, axis=axis, length=length))
        if keepdims == 0 and axis is not None:
            for i in range(len(res)):
                shape = list(res[i].shape)
                del shape[axis]
                res[i] = res[i].reshape(tuple(shape))
        return (res,)
