# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.op_run import OpRun
from onnx.reference.ops._op_common_gated import _check_gated_inputs


class SwiGLU(OpRun):
    def _run(self, a, b, alpha=None):
        _check_gated_inputs("SwiGLU", a, b)
        alpha = 1.0 if alpha is None else alpha
        # alpha scales the sigmoid inside the Swish gate: Swish_alpha(a) = a * sigmoid(alpha * a).
        # Cast the sigmoid term to the input dtype before multiplying, matching the
        # Swish reference implementation's casting behavior.
        swish_a = a * (1 / (1 + np.exp(-alpha * a))).astype(a.dtype)
        return ((swish_a * b).astype(a.dtype),)
