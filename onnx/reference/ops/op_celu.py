# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference import apimod, astype
from onnx.reference.op_run import OpRun


def _vcelu1(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    mod = apimod(x)
    positive_input = mod.maximum(0, x)
    negative_input = mod.minimum(0, alpha * (mod.exp(x / alpha) - 1))
    return positive_input + negative_input  # type: ignore


class Celu(OpRun):
    def _run(self, x, alpha=None):  # type: ignore
        return (astype(_vcelu1(x, alpha), x.dtype),)
