# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import math

import numpy as np

from onnx.reference.ops._op import OpRunUnaryNum


class Gelu(OpRunUnaryNum):
    def _run(self, x, approximate="none"):  # type: ignore
        if approximate == "tanh":
            return (x * 0.5 * (1 + np.tanh((np.sqrt(2/np.pi) * (x + 0.044715 * np.power(x, 3))))),)
        return (0.5 * x * (1 + np.vectorize(math.erf)(x / np.sqrt(2))),)
    
