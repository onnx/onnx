# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.ops._op import OpRunUnaryNum


class LpNormalization(OpRunUnaryNum):
    def _run(self, x, axis=None, p=None):
        axis = self.axis if axis is None else axis
        p = self.p if p is None else p
        norm = np.power(np.power(np.abs(x), p).sum(axis=axis), 1.0 / p)
        norm = np.expand_dims(norm, axis)
        # When norm is 0, return 0 instead of NaN (0/0 = 0)
        result = np.where(norm == 0, 0, x / norm)
        return (result.astype(x.dtype),)
