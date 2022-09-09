# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from ...mapping import TENSOR_TYPE_TO_NP_TYPE
from ..op_run import OpRun


class _CommonWindow(OpRun):
    def __init__(self, onnx_node, run_params):  # type: ignore
        OpRun.__init__(self, onnx_node, run_params)
        self.dtype = TENSOR_TYPE_TO_NP_TYPE[self.output_datatype]

    def _begin(self, size):
        if self.periodic == 1:
            N_1 = size
        else:
            N_1 = size - 1
        ni = numpy.arange(size, dtype=self.dtype)
        return ni, N_1

    def _end(self, size, res):
        return (res.astype(self.dtype),)
