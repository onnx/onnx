# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.op_run import OpRun


class SwiGLU(OpRun):
    def _run(self, a, b, alpha=None):
        alpha = 1.0 if alpha is None else alpha
        # alpha scales the sigmoid inside the Swish gate: Swish_alpha(a) = a * sigmoid(alpha * a).
        swish_a = a * (1 / (1 + np.exp(-alpha * a)))
        return ((swish_a * b).astype(a.dtype),)
