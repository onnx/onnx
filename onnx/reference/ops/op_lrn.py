# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math

import numpy as np

from onnx.reference.op_run import OpRun


class LRN(OpRun):
    def _run(self, x, alpha=None, beta=None, bias=None, size=None):
        if len(x.shape) < 2:
            raise RuntimeError(
                f"LRN expects an input with at least 2 dimensions but shape is {x.shape!r}."
            )
        square_sum = np.zeros_like(x)
        channel_count = x.shape[1]
        c1 = math.floor((size - 1) / 2)
        c2 = math.ceil((size - 1) / 2) + 1
        for c in range(channel_count):
            begin = max(0, c - c1)
            end = min(channel_count, c + c2)
            square_sum[:, c, ...] = np.sum(x[:, begin:end, ...] ** 2, axis=1)
        y = x / ((bias + (alpha / size) * square_sum) ** beta)
        return (y.astype(x.dtype),)
