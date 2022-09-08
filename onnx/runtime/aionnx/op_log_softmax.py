# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from .op_softmax import Softmax


class LogSoftmax(Softmax):
    def __init__(self, onnx_node, run_params):  # type: ignore
        Softmax.__init__(self, onnx_node, run_params)

    def _run(self, X):  # type: ignore
        Y = Softmax._run(self, X)[0]
        numpy.log(Y, out=Y)
        return (Y,)
