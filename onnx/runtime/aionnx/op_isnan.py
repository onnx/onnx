# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from ._op import OpRunUnary


class IsNaN(OpRunUnary):
    def __init__(self, onnx_node, run_params):  # type: ignore
        OpRunUnary.__init__(self, onnx_node, run_params)

    def _run(self, data):  # type: ignore
        return (numpy.isnan(data),)
