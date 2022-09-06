# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

from ._op import OpRunBinaryNum
from ._op_numpy_helper import numpy_matmul


class MatMul(OpRunBinaryNum):
    def __init__(self, onnx_node, run_params):  # type: ignore
        OpRunBinaryNum.__init__(self, onnx_node, run_params)

    def _run(self, a, b):  # type: ignore
        return (numpy_matmul(a, b),)
