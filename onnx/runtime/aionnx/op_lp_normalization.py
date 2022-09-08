# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from ._op import OpRunUnaryNum


class LpNormalization(OpRunUnaryNum):
    def __init__(self, onnx_node, run_params):  # type: ignore
        OpRunUnaryNum.__init__(self, onnx_node, run_params)

    def _run(self, x):  # type: ignore
        norm = numpy.power(numpy.power(x, self.p).sum(axis=self.axis), 1.0 / self.p)  # type: ignore
        norm = numpy.expand_dims(norm, self.axis)  # type: ignore
        return (x / norm,)
