# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.ops._op import OpRunUnaryNum


def sigmoid(x: np.ndarray) -> np.ndarray:
    result = np.empty_like(x)
    pos_mask  = x > 0
    neg_mask = ~pos_mask
    x_neg = x[neg_mask]
    result[pos_mask] = 1 / (1 + np.exp(-x[pos_mask]))
    result[neg_mask] = np.exp(x_neg) / (1 + np.exp(x_neg))
    return result


class Sigmoid(OpRunUnaryNum):
    def __init__(self, onnx_node, run_params):
        OpRunUnaryNum.__init__(self, onnx_node, run_params)

    def _run(self, X):
        return (sigmoid(X),)
