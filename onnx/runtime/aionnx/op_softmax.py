# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from ._op import OpRunUnaryNum


class Softmax(OpRunUnaryNum):
    def __init__(self, onnx_node, run_params):  # type: ignore
        OpRunUnaryNum.__init__(self, onnx_node, run_params)

    def _run(self, X):  # type: ignore
        tmp = X - X.max(axis=self.axis, keepdims=1)  # type: ignore
        Y = numpy.exp(tmp)
        Y /= Y.sum(axis=self.axis, keepdims=1)  # type: ignore
        return (Y,)
