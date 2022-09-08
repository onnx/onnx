# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from ..op_run import OpRun


class Compress(OpRun):
    def __init__(self, onnx_node, run_params):  # type: ignore
        OpRun.__init__(self, onnx_node, run_params)

    def _run(self, x, condition):  # type: ignore
        return (numpy.compress(condition, x, axis=self.axis),)  # type: ignore
