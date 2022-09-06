# SPDX-License-Identifier: Apache-2.0

import numpy  # type: ignore

from ._op import OpRunUnaryNum


class Exp(OpRunUnaryNum):
    def __init__(self, onnx_node, run_params):  # type: ignore
        OpRunUnaryNum.__init__(self, onnx_node, run_params)

    def _run(self, x):  # type: ignore
        return (numpy.exp(x),)
