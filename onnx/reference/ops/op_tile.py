# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from onnx.reference.op_run import OpRun


class Tile(OpRun):
    def _run(self, x, repeats):
        xp = self._get_array_api_namespace(x)
        return (xp.tile(x, repeats),)