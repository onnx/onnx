# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.ops._op import OpRunUnaryNum


class ThresholdedRelu(OpRunUnaryNum):
    def _run(self, x, alpha=None):
        alpha = alpha or self.alpha
        return (np.where(x > alpha, x, 0).astype(x.dtype),)
