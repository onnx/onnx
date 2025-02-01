# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from onnx.reference import astype
from onnx.reference.ops._op import OpRunBinaryNumpy


class Div(OpRunBinaryNumpy):
    def __init__(self, onnx_node, run_params):  # type: ignore
        OpRunBinaryNumpy.__init__(self, (lambda x, y: x / y), onnx_node, run_params)

    def _run(self, a, b):  # type: ignore
        res = OpRunBinaryNumpy._run(self, a, b)
        if res[0].dtype != a.dtype:
            return (astype(res[0], a.dtype),)
        return res
