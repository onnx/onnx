# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from ._op import OpRunUnaryNum


class Sinh(OpRunUnaryNum):
    def __init__(self, onnx_node, run_params):  # type: ignore # noqa: W0221
        OpRunUnaryNum.__init__(self, onnx_node, run_params)

    def _run(self, x):  # type: ignore
        return (numpy.sinh(x),)
