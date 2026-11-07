# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.ops._op import OpRunUnaryNum


class LpNormalization(OpRunUnaryNum):
    def _run(self, x, axis=None, p=None):
        axis = axis or self.axis
        p = p or self.p
        # float16 and bfloat16 slices are accumulated in float32: after scaling,
        # the sum of powers can still exceed their range (e.g. 65520 ones in
        # float16) although the normalized result is representable.
        working_dtype = np.float64 if x.dtype == np.float64 else np.float32
        working = x.astype(working_dtype, copy=False)
        scale = np.max(np.abs(working), axis=axis, keepdims=True, initial=0)
        safe_scale = np.where(
            (scale == 0) | ~np.isfinite(scale),
            np.array(1, dtype=working_dtype),
            scale,
        )
        scaled = working / safe_scale
        norm = np.power(
            np.power(np.abs(scaled), p).sum(axis=axis, keepdims=True), 1.0 / p
        )
        safe_norm = np.where(norm == 0, np.array(1, dtype=working_dtype), norm)
        # When all values along the axis are 0, return 0 instead of NaN.
        result = np.where(scale == 0, 0, scaled / safe_norm)
        return (result.astype(x.dtype),)
