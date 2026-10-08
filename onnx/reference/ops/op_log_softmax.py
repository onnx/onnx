# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.ops.op_softmax import Softmax


class LogSoftmax(Softmax):
    def _run(self, X):
        if X.size == 0:
            return (X,)
        shape = X.shape
        axis = self.axis
        if self.run_params["opsets"][""] < 13:
            # Legacy LogSoftmax groups all dimensions from axis onward as features.
            if not any(att.name == "axis" for att in self.onnx_node.attribute):
                axis = 1
            axis %= X.ndim
            X = X.reshape((int(np.prod(shape[:axis])), -1))
            axis = 1
        tmp = X - X.max(axis=axis, keepdims=True)
        Y = tmp - np.log(np.exp(tmp).sum(axis=axis, keepdims=True))
        Y = Y.astype(X.dtype)
        return (Y.reshape(shape),)
