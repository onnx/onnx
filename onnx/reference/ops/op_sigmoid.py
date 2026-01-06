# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.ops._op import OpRunUnaryNum


def sigmoid(x: np.ndarray) -> np.ndarray:
    result = np.empty_like(x)
    gt_zero = x > 0
    le_zero = ~gt_zero
    result[gt_zero] = 1 / (1 + np.exp(-x[gt_zero]))
    result[le_zero] = np.exp(x[le_zero]) / (1 + np.exp(x[le_zero]))
    return result


class Sigmoid(OpRunUnaryNum):
    def __init__(self, onnx_node, run_params):
        OpRunUnaryNum.__init__(self, onnx_node, run_params)

    def _run(self, X):
        return (sigmoid(X),)
