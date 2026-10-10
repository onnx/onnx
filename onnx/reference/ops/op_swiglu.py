# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.op_run import OpRun


class SwiGLU(OpRun):
    def _run(self, a, b, alpha=None):
        # SwiGLU requires identical shapes and dtypes for A and B: broadcasting is
        # not applied, matching the equal-shape/no-broadcast contract enforced by
        # SwiGLUShapeInference at graph-build time.
        if a.shape != b.shape:
            raise ValueError(
                "SwiGLU requires inputs A and B to have identical shapes "
                f"(broadcasting is not applied), but got A.shape={a.shape} and "
                f"B.shape={b.shape}."
            )
        if a.dtype != b.dtype:
            raise ValueError(
                "SwiGLU requires inputs A and B to have identical dtypes, but "
                f"got A.dtype={a.dtype} and B.dtype={b.dtype}."
            )
        alpha = 1.0 if alpha is None else alpha
        # alpha scales the sigmoid inside the Swish gate: Swish_alpha(a) = a * sigmoid(alpha * a).
        # Cast the sigmoid term to the input dtype before multiplying, matching the
        # Swish reference implementation's casting behavior.
        swish_a = a * (1 / (1 + np.exp(-alpha * a))).astype(a.dtype)
        return ((swish_a * b).astype(a.dtype),)
