# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.ops._op import OpRunUnaryNum


def _run_on_2d(run, X, axis):
    """Runs ``run`` on the last axis of X coerced to 2D at ``axis``, as opsets before 13 do."""
    if X.size == 0:
        return (X,)
    X_2d = X.reshape(int(np.prod(X.shape[:axis])), -1)
    return (run(X_2d, axis=1)[0].reshape(X.shape),)


class Softmax_13(OpRunUnaryNum):
    def _run(self, X, axis=None):
        if X.size == 0:
            return (X,)
        axis = axis or self.axis
        tmp = X - X.max(axis=axis, keepdims=1)
        Y = np.exp(tmp)
        Y /= Y.sum(axis=axis, keepdims=1)
        return (Y.astype(X.dtype),)


class Softmax_1(Softmax_13):
    def _run(self, X, axis=None):
        return _run_on_2d(super()._run, X, self.axis if axis is None else axis)
