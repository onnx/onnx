# SPDX-License-Identifier: Apache-2.0
# pylint: disable=R0913,W0221

import math

import numpy as np

from onnx.reference.op_run import OpRun


class LRN(OpRun):
    def _run(self, x, alpha=None, beta=None, bias=None, size=None):  # type: ignore
        if len(x.shape) != 4:
            raise RuntimeError(
                f"LRN only applies on 4D tensors but shape is {x.shape!r}."
            )
        square_sum = np.zeros(x.shape).astype(x.dtype)
        for ind in np.ndindex(x.shape):
            n, c, h, w = ind
            begin = max(0, c - int(math.floor((size - 1) / 2)))
            end = min(5, c + int(math.ceil((size - 1) / 2)) + 1)
            square_sum[n, c, h, w] = np.sum(x[n, begin:end, h, w] ** 2)
        y = x / ((bias + (alpha / size) * square_sum) ** beta)
        return (y.astype(x.dtype),)
