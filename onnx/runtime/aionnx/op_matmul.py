# SPDX-License-Identifier: Apache-2.0

from ._op import OpRunBinaryNum
from ._op_numpy_helper import numpy_matmul


class MatMul(OpRunBinaryNum):
    def __init__(self, onnx_node, run_params):  # type: ignore
        OpRunBinaryNum.__init__(self, onnx_node, run_params)

    def _run(self, a, b, attributes=None):  # type: ignore # pylint: disable=W0221
        return (numpy_matmul(a, b),)
