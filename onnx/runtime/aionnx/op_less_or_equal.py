# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from ._op import OpRunBinaryComparison


class LessOrEqual(OpRunBinaryComparison):
    def __init__(self, onnx_node, run_params):  # type: ignore
        OpRunBinaryComparison.__init__(self, onnx_node, run_params)

    def _run(self, a, b):  # type: ignore
        return (numpy.less_equal(a, b),)
