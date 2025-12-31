# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from onnx.reference.ops._op import OpRunBinaryNumpy


class Add(OpRunBinaryNumpy):
    def __init__(self, onnx_node, run_params):
        # Store a lambda that uses array API add
        OpRunBinaryNumpy.__init__(
            self, lambda a, b: self._get_array_api_namespace(a, b).add(a, b), onnx_node, run_params
        )
