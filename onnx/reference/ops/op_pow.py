# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from warnings import catch_warnings, simplefilter

from onnx.reference.op_run import OpRun


class Pow(OpRun):
    def _run(self, a, b):
        xp = self._get_array_api_namespace(a, b)
        with catch_warnings():
            simplefilter("ignore")
            return (xp.pow(a, b),)
