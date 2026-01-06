# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.ops._op import OpRunUnaryNum


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Numerically stable sigmoid implementation that supports scalars
    (0-dimensional arrays) and higher-dimensional arrays.
    """
    pos_mask = x > 0
    return np.where(
        pos_mask,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


class Sigmoid(OpRunUnaryNum):
    def __init__(self, onnx_node, run_params):
        OpRunUnaryNum.__init__(self, onnx_node, run_params)

    def _run(self, X):
        return (sigmoid(X),)
