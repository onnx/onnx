# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from onnx.reference import apimod
from onnx.reference.ops._op import OpRunBinary


class And(OpRunBinary):
    def _run(self, x, y):  # type: ignore
        return (apimod(x).logical_and(x, y),)
