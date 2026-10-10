# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math

import numpy as np

from onnx.reference.op_run import OpRun


class GlobalLpPool(OpRun):
    def _run(self, x, p=2):
        spatial_axes = tuple(range(2, x.ndim))
        if any(x.shape[axis] == 0 for axis in spatial_axes):
            output_shape = (*x.shape[:2], *(1 for _ in spatial_axes))
            return (np.zeros(output_shape, dtype=x.dtype),)

        p = float(p)
        reciprocal_p = 1.0 / p
        spatial_size = math.prod(x.shape[2:])
        values = np.abs(x.astype(np.float64)).reshape((*x.shape[:2], spatial_size))
        with np.errstate(
            divide="ignore", invalid="ignore", over="ignore", under="ignore"
        ):
            powered_logs = p * np.log(values)
            log_power_sum = np.logaddexp.reduce(powered_logs, axis=-1, keepdims=True)
            norm = np.exp(log_power_sum * reciprocal_p)
        output_shape = (*x.shape[:2], *(1 for _ in spatial_axes))
        norm = norm.reshape(output_shape)
        return (norm.astype(x.dtype),)
