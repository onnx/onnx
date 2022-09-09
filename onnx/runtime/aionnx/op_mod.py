# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from ..op_run import OpRun


class Mod(OpRun):

    atts = {"fmod": 0}

    def __init__(self, onnx_node, run_params):  # type: ignore
        OpRun.__init__(self, onnx_node, run_params)

    def _run(self, a, b):  # type: ignore
        if self.fmod == 1:  # type: ignore
            return (numpy.fmod(a, b),)
        if a.dtype in (numpy.float16, numpy.float32, numpy.float64):
            return (numpy.nan_to_num(numpy.fmod(a, b)),)
        return (numpy.nan_to_num(numpy.mod(a, b)),)
