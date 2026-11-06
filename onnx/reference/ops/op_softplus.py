# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.ops._op import OpRunUnaryNum


class Softplus(OpRunUnaryNum):
    def _run(self, X):
        return (np.logaddexp(0, X).astype(X.dtype),)
