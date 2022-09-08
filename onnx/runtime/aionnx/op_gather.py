# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from ..op_run import OpRun


class Gather(OpRun):
    def __init__(self, onnx_node, run_params):  # type: ignore
        OpRun.__init__(self, onnx_node, run_params)

    def _run(self, x, indices):  # type: ignore
        if not x.flags["C_CONTIGUOUS"]:
            x = numpy.ascontiguousarray(x)
        if not indices.flags["C_CONTIGUOUS"]:
            indices = indices.ascontiguousarray()
        if indices.size == 0:
            return (numpy.empty((0,), dtype=x.dtype),)
        return (numpy.take(x, indices, axis=self.axis),)  # type: ignore
