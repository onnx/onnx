# SPDX-License-Identifier: Apache-2.0

import numpy  # type: ignore

from ._op import OpRunBinaryNumpy


class Add(OpRunBinaryNumpy):
    def __init__(self, onnx_node, log_function):  # type: ignore # noqa: W0221
        OpRunBinaryNumpy.__init__(self, numpy.add, onnx_node, log_function)
