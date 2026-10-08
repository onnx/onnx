# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math

import numpy as np

from onnx.reference.ops._op import OpRunUnaryNum


class Softmax(OpRunUnaryNum):
    def _run(self, X, axis=None):
        if X.size == 0:
            return (X,)
        shape = X.shape
        legacy = self.run_params["opsets"][""] < 13
        if (
            legacy
            and axis is None
            and not any(
                attribute.name == "axis" for attribute in self.onnx_node.attribute
            )
        ):
            # The unversioned class receives the latest schema's default axis.
            # Earlier models default to axis1 before coercing the input to 2D.
            axis = 1
        axis = axis or self.axis
        if legacy:
            axis %= X.ndim
            X = X.reshape((-1, math.prod(shape[axis:])))
            axis = 1
        tmp = X - X.max(axis=axis, keepdims=1)
        Y = np.exp(tmp)
        Y /= Y.sum(axis=axis, keepdims=1)
        return (Y.astype(X.dtype).reshape(shape),)
