# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.ops._op import OpRunUnaryNum


class Swish(OpRunUnaryNum):
    def _run(self, x, alpha=None):
        alpha = self.alpha if alpha is None else alpha
        return (x * (1 / (1 + np.exp(-alpha * x))).astype(x.dtype),)
