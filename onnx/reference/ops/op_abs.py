# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from onnx.reference import apimod
from onnx.reference.ops._op import OpRunUnaryNum


class Abs(OpRunUnaryNum):
    def _run(self, x):  # type: ignore
        return (apimod(x).absolute(x),)
