# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

from math import erf

import numpy  # type: ignore

from ._op import OpRunUnaryNum


class Erf(OpRunUnaryNum):
    def __init__(self, onnx_node, run_params):  # type: ignore
        OpRunUnaryNum.__init__(self, onnx_node, run_params)
        self._erf = numpy.vectorize(erf)

    def _run(self, x):  # type: ignore
        return (self._erf(x).astype(x.dtype),)
