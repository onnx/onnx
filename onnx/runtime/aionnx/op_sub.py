# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy as np  # type: ignore

from ._op import OpRunBinaryNumpy


class Sub(OpRunBinaryNumpy):
    def __init__(self, onnx_node, run_params):  # type: ignore
        OpRunBinarynp.__init__(self, np.subtract, onnx_node, run_params)
