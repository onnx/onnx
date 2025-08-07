# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.ops._op import OpRunUnaryNum


class HardSigmoid(OpRunUnaryNum):
    def _run(self, x, alpha=None, beta=None):
        alpha = alpha or self.alpha
        beta = beta or self.beta
        y = np.maximum(0, np.minimum(1, x * alpha + beta)).astype(x.dtype)
        return (y,)
