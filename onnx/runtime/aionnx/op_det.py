# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from ..op_run import OpRun


class Det(OpRun):
    def __init__(self, onnx_node, run_params):  # type: ignore
        OpRun.__init__(self, onnx_node, run_params)

    def _run(self, x):  # type: ignore
        res = numpy.linalg.det(x)
        if not isinstance(res, numpy.ndarray):
            res = numpy.array([res])
        return (res,)
