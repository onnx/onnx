# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from onnx.reference.ops._op import OpRunBinaryNumpy


class Add(OpRunBinaryNumpy):
    def __init__(self, onnx_node, run_params):  # type: ignore
        OpRunBinaryNumpy.__init__(self, (lambda x, y: x + y), onnx_node, run_params)
