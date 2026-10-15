# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math

import ml_dtypes
import numpy as np

from onnx.reference.op_run import OpRun


class LRN(OpRun):
    def _run(self, x, alpha=None, beta=None, bias=None, size=None):
        minimum_rank = 2
        if len(x.shape) < minimum_rank:
            raise RuntimeError(
                f"LRN expects an input with at least 2 dimensions but shape is {x.shape!r}."
            )
        # float16 and bfloat16 are computed in float32: the sum of squares
        # overflows float16 (max 65504) for values above ~256.
        xc = x.astype(np.float32) if x.dtype in (np.float16, ml_dtypes.bfloat16) else x
        square_sum = np.zeros_like(xc)
        channel_count = x.shape[1]
        c1 = math.floor((size - 1) / 2)
        c2 = math.ceil((size - 1) / 2) + 1
        for c in range(channel_count):
            begin = max(0, c - c1)
            end = min(channel_count, c + c2)
            square_sum[:, c, ...] = np.sum(xc[:, begin:end, ...] ** 2, axis=1)
        y = xc / ((bias + (alpha / size) * square_sum) ** beta)
        return (y.astype(x.dtype),)
