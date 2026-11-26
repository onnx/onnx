# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ml_dtypes
import numpy as np

from onnx.reference.op_run import OpRun


class InstanceNormalization(OpRun):
    def _run(self, x, s, bias, epsilon=None):
        dims_x = len(x.shape)
        axis = tuple(range(2, dims_x))
        # float16 and bfloat16 are computed in float32: the variance squares
        # the input and overflows float16 (max 65504) for values above ~256.
        xc = x.astype(np.float32) if x.dtype in (np.float16, ml_dtypes.bfloat16) else x
        mean = np.mean(xc, axis=axis, keepdims=True)
        var = np.var(xc, axis=axis, keepdims=True)
        dim_ones = (1,) * (dims_x - 2)
        s = s.reshape(-1, *dim_ones)
        bias = bias.reshape(-1, *dim_ones)
        y = s * (xc - mean) / np.sqrt(var + epsilon) + bias
        return (y.astype(x.dtype),)
