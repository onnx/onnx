# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.ops.aionnxml._op_run_aionnxml import OpRunAiOnnxMl


class LinearRegressor(OpRunAiOnnxMl):
    def _run(
        self, x, coefficients=None, intercepts=None, targets=1, post_transform=None
    ):
        # The output is always tensor(float), also for integer and double inputs.
        x = x.astype(np.float32)
        coefficients = np.array(coefficients, dtype=np.float32)
        intercepts = np.array(intercepts, dtype=np.float32)
        n = coefficients.shape[0] // targets
        coefficients = coefficients.reshape(targets, n).T
        score = np.dot(x, coefficients)
        if intercepts is not None:
            score += intercepts
        if post_transform == "NONE":
            return (score,)
        raise NotImplementedError(
            f"post_transform: {post_transform!r} is not implemented."
        )
