# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ml_dtypes
import numpy as np

from onnx.reference.ops.op_softmax import Softmax


class LogSoftmax(Softmax):
    def _run(self, X):
        if X.size == 0:
            return (X,)
        # float16 and bfloat16 are computed in float32: the sum of exp over the
        # axis overflows float16 (max 65504) once the axis has more elements
        # than that, and bfloat16 stops accumulating at 256.
        Xc = X.astype(np.float32) if X.dtype in (np.float16, ml_dtypes.bfloat16) else X
        tmp = Xc - Xc.max(axis=self.axis, keepdims=True)
        Y = tmp - np.log(np.exp(tmp).sum(axis=self.axis, keepdims=True))
        Y = Y.astype(X.dtype)
        return (Y,)
