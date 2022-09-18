# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

from ...defs import onnx_opset_version
from ..op_run import OpRun


class CommonSplit(OpRun):
    def __init__(self, onnx_node, run_params):  # type: ignore
        OpRun.__init__(self, onnx_node, run_params)
        self.n_outputs = len(onnx_node.output)

    def common_run(self, mat, split):  # type: ignore
        if split is None:
            div = mat.shape[self.axis] // self.n_outputs  # type: ignore
            split = [div] * self.n_outputs
            split[-1] += mat.shape[self.axis] - sum(split)  # type: ignore
        sli = [slice(0, s) for s in mat.shape]
        res = []
        pos = 0
        for spl in split:
            sli[self.axis] = slice(pos, pos + spl)  # type: ignore
            pos += spl
            res.append(mat[tuple(sli)])
        return tuple(res)


class Split_2(CommonSplit):
    def _run(self, mat):  # type: ignore
        # TODO: support overridden attributes.
        return self.common_run(mat, self.split)  # type: ignore


class Split_11(Split_2):
    pass


class Split_13(CommonSplit):
    def _run(self, mat, split=None):  # type: ignore
        # TODO: support overridden attributes.
        return self.common_run(mat, split)


if onnx_opset_version() >= 13:
    Split = Split_13
elif onnx_opset_version() >= 11:
    Split = Split_11  # type: ignore
else:
    Split = Split_2  # type: ignore
