# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from ._op import OpRunUnary


class IsInf(OpRunUnary):
    def __init__(self, onnx_node, run_params):  # type: ignore
        OpRunUnary.__init__(self, onnx_node, run_params)

    def _run(self, data):  # type: ignore
        if self.detect_negative:  # type: ignore
            if self.detect_positive:  # type: ignore
                return (numpy.isinf(data),)
            return (numpy.isneginf(data),)
        if self.detect_positive:  # type: ignore
            return (numpy.isposinf(data),)
        res = numpy.full(data.shape, dtype=numpy.bool_, fill_value=False)
        return (res,)
