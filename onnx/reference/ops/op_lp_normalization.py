# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ml_dtypes
import numpy as np

from onnx.reference.ops._op import OpRunUnaryNum


class LpNormalization(OpRunUnaryNum):
    def _run(self, x, axis=None, p=None):
        axis = axis or self.axis
        p = p or self.p
        # float16 and bfloat16 are computed in float32: raising the input to
        # the power p overflows float16 (max 65504) for values above 256
        # when p is 2.
        xc = x.astype(np.float32) if x.dtype in (np.float16, ml_dtypes.bfloat16) else x
        norm = np.power(np.power(np.abs(xc), p).sum(axis=axis), 1.0 / p)
        norm = np.expand_dims(norm, axis)
        # When norm is 0, return 0 instead of NaN (0/0 = 0)
        result = np.where(norm == 0, 0, xc / norm)
        return (result.astype(x.dtype),)
