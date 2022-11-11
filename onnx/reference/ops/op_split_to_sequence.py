# SPDX-License-Identifier: Apache-2.0
# pylint: disable=C0200,W0221

from .op_split import CommonSplit


class SplitToSequence(CommonSplit):
    def _run(self, mat, split=None, axis=None, keepdims=None):  # type: ignore
        if split is None:
            num_outputs = mat.shape[axis]
        else:
            if len(split.shape) == 0:
                num_outputs = split
            else:
                num_outputs = split.shape[0]
        res = list(self.common_run(mat, split, axis=axis, num_outputs=num_outputs))
        if keepdims == 0 and axis is not None:
            for i in range(len(res)):
                shape = list(res[i].shape)
                del shape[axis]
                res[i] = res[i].reshape(tuple(shape))
        return (res,)
