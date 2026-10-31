# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.op_run import OpRun


class DynamicQuantizeLinear(OpRun):
    def _run(self, x):
        # args: x, y_scale, zero_point
        dtype, qmin, qmax = np.uint8, 0, 255
        maxx = np.float32(np.maximum(0, np.max(x)))
        minx = np.float32(np.minimum(0, np.min(x)))
        # The adjusted range always contains 0, so maxx == minx only when every
        # element of x is 0. The range is then empty and the scale is defined
        # to be 1 (which gives y_zero_point = 0 and y = 0) instead of 0.
        if maxx == minx:
            y_scale = np.float32(1.0)
        else:
            y_scale = (maxx - minx) / np.float32(qmax - qmin)

        initial_zero_point = np.float32(qmin) - minx / y_scale
        zp = max(qmin, min(qmax, initial_zero_point))
        zpi = np.rint(zp)

        y = np.clip(np.rint(x / y_scale) + zpi, qmin, qmax)
        return (
            y.astype(dtype),
            np.array(y_scale.astype(x.dtype)),
            np.array(zpi.astype(dtype)),
        )
