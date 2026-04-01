# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

from onnx.reference.ops._op import OpRunUnaryNum

if TYPE_CHECKING:
    import numpy as np


def _leaky_relu(x: np.ndarray, alpha: float) -> np.ndarray:
    sign = (x > 0).astype(x.dtype)
    sign -= ((sign - 1) * alpha).astype(x.dtype)
    return x * sign


class LeakyRelu(OpRunUnaryNum):
    def _run(self, x, alpha=None):
        alpha = alpha or self.alpha
        return (_leaky_relu(x, alpha).astype(x.dtype),)
