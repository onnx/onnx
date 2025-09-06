# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.ops._op import OpRunBinaryNumpy


class BitShift(OpRunBinaryNumpy):
    def __init__(self, onnx_node, run_params):
        OpRunBinaryNumpy.__init__(self, np.right_shift, onnx_node, run_params)
        if self.direction not in ("LEFT", "RIGHT"):
            raise ValueError(f"Unexpected value for direction ({self.direction!r}).")
        if self.direction == "LEFT":
            self.numpy_fct = np.left_shift
