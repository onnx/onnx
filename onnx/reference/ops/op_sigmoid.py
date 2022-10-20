# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy as np

from ._op import OpRunUnaryNum


def sigmoid(x):  # type: ignore
    if x > 0:
        return 1 / (1 + np.exp(-x))
    return np.exp(x) / (1 + np.exp(x))


class Sigmoid(OpRunUnaryNum):
    def __init__(self, onnx_node, run_params):  # type: ignore
        OpRunUnaryNum.__init__(self, onnx_node, run_params)
        self.vf = np.vectorize(sigmoid)

    def _run(self, X):  # type: ignore
        return (self.vf(X).astype(X.dtype),)
