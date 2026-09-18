# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.ops._op import OpRunUnaryNum


class LpNormalization(OpRunUnaryNum):
    def _run(self, x, axis=None, p=None):
        axis = axis or self.axis
        p = p or self.p
        scale = np.max(np.abs(x), axis=axis, keepdims=True, initial=0)
        safe_scale = np.where(
            (scale == 0) | ~np.isfinite(scale),
            np.array(1, dtype=x.dtype),
            scale,
        )
        scaled = x / safe_scale
        norm = np.power(
            np.power(np.abs(scaled), p).sum(axis=axis, keepdims=True), 1.0 / p
        )
        safe_norm = np.where(norm == 0, np.array(1, dtype=x.dtype), norm)
        # When all values along the axis are 0, return 0 instead of NaN.
        result = np.where(scale == 0, 0, scaled / safe_norm)
        return (result.astype(x.dtype),)
