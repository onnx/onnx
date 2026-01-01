# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

from onnx.reference.op_run import OpRun


class SplitToSequence(OpRun):
    def common_run(
        self, mat: Any, split: Any | None, axis: int
    ) -> list[Any]:
        if split is None:
            split_length = [1 for _ in range(mat.shape[axis])]
        elif len(split.shape) == 0:
            # A scalar
            dim = mat.shape[axis]
            length = int(split)
            n = dim // int(length)
            split_length = [length] * n
            left = dim - length * n
            if left > 0:
                split_length.append(left)
        else:
            split_length = list(split)

        sli = [slice(0, s) for s in mat.shape]
        res = []
        pos = 0
        for spl in split_length:
            sli[axis] = slice(pos, pos + spl)
            pos += spl
            res.append(mat[tuple(sli)])
        return res

    def _run(
        self,
        mat: Any,
        split: Any | None = None,
        axis: int = 0,
        keepdims: int = 1,
    ) -> tuple[Any]:
        res = self.common_run(mat, split, axis=axis)
        if split is None and not keepdims:
            for i, res_i in enumerate(res):
                shape = list(res_i.shape)
                del shape[axis]
                res[i] = res_i.reshape(tuple(shape))
        return (res,)