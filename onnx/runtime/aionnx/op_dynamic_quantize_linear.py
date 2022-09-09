# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from ..op_run import OpRun


class DynamicQuantizeLinear(OpRun):
    def __init__(self, onnx_node, run_params):  # type: ignore
        OpRun.__init__(self, onnx_node, run_params)
        self.dtype = numpy.uint8

    def _run(self, x):  # type: ignore
        # args: x, y_scale, zero_point
        qmin, qmax = 0, 255
        maxx = numpy.maximum(0, numpy.max(x))
        minx = numpy.minimum(0, numpy.min(x))
        y_scale = (maxx - minx) / (qmax - qmin)
        intermediate_zero_point = numpy.round(qmin - minx) / y_scale
        y_zero_point = numpy.round(
            numpy.clip(intermediate_zero_point, qmin, qmax)
        ).astype(self.dtype)
        y = numpy.clip(numpy.round(x / y_scale) + y_zero_point, qmin, qmax)
        return (
            y.astype(self.dtype),
            y_scale.astype(x.dtype),
            y_zero_point.astype(self.dtype),
        )
