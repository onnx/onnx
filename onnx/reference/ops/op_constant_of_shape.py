# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.op_run import OpRun


class ConstantOfShape(OpRun):
    def __init__(self, onnx_node, run_params):
        OpRun.__init__(self, onnx_node, run_params)
        self.cst = self.value[0] if isinstance(self.value, np.ndarray) else self.value
        if isinstance(self.cst, int):
            self.cst = np.int64(self.cst)
        elif isinstance(self.cst, float):
            self.cst = np.float64(self.cst)
        elif self.cst is None:
            self.cst = np.float32(0)

    def _run(self, data, value=None):
        try:
            res = np.full(tuple(data), self.cst)
        except TypeError as e:
            raise RuntimeError(
                f"Unable to create a constant of shape {data!r} with value {self.cst!r} "
                f"(raw value={value!r})."
            ) from e
        return (res,)
