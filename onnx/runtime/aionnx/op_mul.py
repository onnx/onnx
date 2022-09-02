# SPDX-License-Identifier: Apache-2.0

import numpy
from ._op import OpRunBinaryNumpy


class Mul(OpRunBinaryNumpy):
    def __init__(self, onnx_node, log_function):
        OpRunBinaryNumpy.__init__(self, numpy.multiply, onnx_node, log_function)
