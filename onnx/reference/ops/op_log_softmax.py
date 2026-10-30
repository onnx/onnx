# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.ops.op_softmax import Softmax_13, _run_on_2d


class LogSoftmax_13(Softmax_13):
    def _run(self, X, axis=None):
        if X.size == 0:
            return (X,)
        axis = self.axis if axis is None else axis
        tmp = X - X.max(axis=axis, keepdims=True)
        Y = tmp - np.log(np.exp(tmp).sum(axis=axis, keepdims=True))
        Y = Y.astype(X.dtype)
        return (Y,)


class LogSoftmax_1(LogSoftmax_13):
    def _run(self, X, axis=None):
        return _run_on_2d(super()._run, X, self.axis if axis is None else axis)
