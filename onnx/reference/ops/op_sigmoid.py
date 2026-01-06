# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.ops._op import OpRunUnaryNum


def sigmoid(x: np.ndarray) -> np.ndarray:
    result = np.empty_like(x)
    result[x > 0] = 1 / (1 + np.exp(-x[x > 0]))
    result[x <= 0] = np.exp(x[x <= 0]) / (1 + np.exp(x[x <= 0]))
    return result


class Sigmoid(OpRunUnaryNum):
    def __init__(self, onnx_node, run_params):
        OpRunUnaryNum.__init__(self, onnx_node, run_params)

    def _run(self, X):
        return (sigmoid(X),)
