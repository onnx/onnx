# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.ops.op_softmax import Softmax


class LogSoftmax(Softmax):
    def _run(self, X, axis=None):
        if X.size == 0:
            return (X,)
        axis = self.axis if axis is None else axis
        tmp = X - X.max(axis=axis, keepdims=True)
        Y = tmp - np.log(np.exp(tmp).sum(axis=axis, keepdims=True))
        Y = Y.astype(X.dtype)
        return (Y,)
