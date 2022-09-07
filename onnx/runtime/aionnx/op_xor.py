# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from ._op import OpRunBinary


class Xor(OpRunBinary):
    def __init__(self, onnx_node, run_params):  # type: ignore
        OpRunBinary.__init__(self, onnx_node, run_params)

    def _run(self, x, y):  # type: ignore
        return (numpy.logical_xor(x, y),)
