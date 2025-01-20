# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.op_run import OpRun
from onnx.reference.ops.op_rms_normalization import _rms_normalization


def _skip_rms_normalization(
    X: np.ndarray,
    S: np.ndarray,
    gamma: np.ndarray,
    B: None | np.ndarray,
    axis: int = -1,
    epsilon: float = 1e-5,
    scaling_factor: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    input_skip_sum = X + (S * scaling_factor)
    if B is not None:
        input_skip_bias_sum = input_skip_sum + B
    else:
        input_skip_bias_sum = input_skip_sum
    output = _rms_normalization(input_skip_bias_sum, gamma, epsilon=epsilon, axis=axis)
    return output, input_skip_bias_sum


class SkipRMSNormalization(OpRun):
    def _run(self, X, S, Scale, B=None, axis=None, epsilon=None, scaling_factor=None):  # type: ignore
        res = _skip_rms_normalization(
            X, S, Scale, B, axis=axis, epsilon=epsilon, scaling_factor=scaling_factor
        )
        return res  # type: ignore
