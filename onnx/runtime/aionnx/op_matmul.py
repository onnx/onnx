# SPDX-License-Identifier: Apache-2.0

from ._op import OpRunBinaryNum
from ._op_numpy_helper import numpy_matmul


class MatMul(OpRunBinaryNum):
    def __init__(self, onnx_node, log_function):
        OpRunBinaryNum.__init__(self, onnx_node, log_function)

    def _run(self, a, b, attributes=None):
        return (numpy_matmul(a, b),)
