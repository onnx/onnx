# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.ops.op_softmax import Softmax


class LogSoftmax(Softmax):
    def _run(self, X):
        if X.size == 0:
            return (X,)
        tmp = X - X.max(axis=self.axis, keepdims=True)
        Y = tmp - np.log(np.exp(tmp).sum(axis=self.axis, keepdims=True))
        Y = Y.astype(X.dtype)
        return (Y,)
