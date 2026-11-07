# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ml_dtypes
import numpy as np

from onnx.reference.ops._op import OpRunUnaryNum


class Softmax(OpRunUnaryNum):
    def _run(self, X, axis=None):
        if X.size == 0:
            return (X,)
        axis = axis or self.axis
        data = (
            X.astype(np.float32) if X.dtype in (np.float16, ml_dtypes.bfloat16) else X
        )
        tmp = data - data.max(axis=axis, keepdims=1)
        Y = np.exp(tmp)
        Y /= Y.sum(axis=axis, keepdims=1)
        return (Y.astype(X.dtype),)
