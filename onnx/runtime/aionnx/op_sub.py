# SPDX-License-Identifier: Apache-2.0

import numpy  # type: ignore

from ._op import OpRunBinaryNumpy


class Sub(OpRunBinaryNumpy):
    def __init__(self, onnx_node, run_params):  # type: ignore
        OpRunBinaryNumpy.__init__(self, numpy.subtract, onnx_node, run_params)
