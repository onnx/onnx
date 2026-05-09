# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.ops._op import OpRunUnaryNum


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid implementation that supports scalars and nd-arrays."""
    pos_mask = x > 0
    exp_x = np.exp(x)
    return np.where(
        pos_mask,
        1.0 / (1.0 + np.exp(-x)),
        exp_x / (1.0 + exp_x),
    )


class Sigmoid(OpRunUnaryNum):
    def __init__(self, onnx_node, run_params):
        OpRunUnaryNum.__init__(self, onnx_node, run_params)

    def _run(self, X):
        return (sigmoid(X).astype(X.dtype),)
