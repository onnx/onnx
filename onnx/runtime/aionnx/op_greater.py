# SPDX-License-Identifier: Apache-2.0

import numpy  # type: ignore

from ._op import OpRunBinaryComparison


class Greater(OpRunBinaryComparison):
    def __init__(self, onnx_node, run_params):  # type: ignore
        OpRunBinaryComparison.__init__(self, onnx_node, run_params)

    def _run(self, a, b, attributes=None):  # type: ignore
        return (numpy.greater(a, b),)


class GreaterOrEqual(OpRunBinaryComparison):
    def __init__(self, onnx_node, run_params):  # type: ignore
        OpRunBinaryComparison.__init__(self, onnx_node, run_params)

    def _run(self, a, b, attributes=None):  # type: ignore
        return (numpy.greater_equal(a, b),)
