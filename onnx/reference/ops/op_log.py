# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from onnx.reference import apimod, astype
from onnx.reference.ops._op import OpRunUnaryNum


class Log(OpRunUnaryNum):
    def _run(self, x):  # type: ignore
        return (astype(apimod(x).log(x), x.dtype),)
