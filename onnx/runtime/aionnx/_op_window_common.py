# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0613,W0221

import numpy as np  # type: ignore

from ...mapping import TENSOR_TYPE_MAP
from ..op_run import OpRun


class _CommonWindow(OpRun):
    def __init__(self, onnx_node, run_params):  # type: ignore
        OpRun.__init__(self, onnx_node, run_params)
        self.dtype = TENSOR_TYPE_MAP[self.output_datatype].np_dtype  # type: ignore

    def _begin(self, size):  # type: ignore
        if self.periodic == 1:  # type: ignore
            N_1 = size
        else:
            N_1 = size - 1
        ni = np.arange(size, dtype=self.dtype)
        return ni, N_1

    def _end(self, size, res):  # type: ignore
        return (res.astype(self.dtype),)
