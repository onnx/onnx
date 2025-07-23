# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.ops._op import OpRunUnaryNum


def sigmoid(x):
    if x > 0:
        return 1 / (1 + np.exp(-x))
    return np.exp(x) / (1 + np.exp(x))


class Sigmoid(OpRunUnaryNum):
    def __init__(self, onnx_node, run_params):
        OpRunUnaryNum.__init__(self, onnx_node, run_params)
        self.vf = np.vectorize(sigmoid)

    def _run(self, X):
        if len(X.shape) == 0:
            return (sigmoid(X).astype(X.dtype),)
        if X.size == 0:
            return (X,)
        return (self.vf(X).astype(X.dtype),)
