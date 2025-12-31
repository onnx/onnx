# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from warnings import catch_warnings, simplefilter

from onnx.reference.ops._op import OpRunUnaryNum


class Sqrt(OpRunUnaryNum):
    def _run(self, x):
        xp = self._get_array_api_namespace(x)
        with catch_warnings():
            simplefilter("ignore")
            return (xp.sqrt(x),)
