# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from ._op import OpRunUnaryNum


class Elu(OpRunUnaryNum):

    atts = {"alpha": 1}

    def __init__(self, onnx_node, run_params):  # type: ignore
        OpRunUnaryNum.__init__(self, onnx_node, run_params)

    def _run(self, x):  # type: ignore
        return (numpy.where(x > 0, x, self.alpha * (numpy.exp(x) - 1)),)  # type: ignore
