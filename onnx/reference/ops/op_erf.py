# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from math import erf

import numpy as np

from onnx.reference.ops._op import OpRunUnaryNum


class Erf(OpRunUnaryNum):
    def __init__(self, onnx_node, run_params):
        OpRunUnaryNum.__init__(self, onnx_node, run_params)
        self._erf = np.vectorize(erf, otypes=["f"])

    def _run(self, x):
        return (self._erf(x).astype(x.dtype),)
