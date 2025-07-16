# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.ops._op import OpRunUnaryNum


class LpNormalization(OpRunUnaryNum):
    def _run(self, x, axis=None, p=None):
        axis = axis or self.axis
        p = p or self.p
        norm = np.power(np.power(x, p).sum(axis=axis), 1.0 / p)
        norm = np.expand_dims(norm, axis)
        return ((x / norm).astype(x.dtype),)
