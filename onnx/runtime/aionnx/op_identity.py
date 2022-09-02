# SPDX-License-Identifier: Apache-2.0

from ._op import OpRunUnaryNum


class Identity(OpRunUnaryNum):

    def __init__(self, onnx_node, log_function):
        OpRunUnaryNum.__init__(self, onnx_node, log_function)

    def _run(self, a, attributes=None):
        if a is None:
            return (None, )
        return (a.copy(), )
