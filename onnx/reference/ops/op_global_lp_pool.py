# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.op_run import OpRun


class GlobalLpPool(OpRun):
    def _run(self, x, p=2):
        spatial_axes = tuple(range(2, x.ndim))
        powered = np.power(np.abs(x), p)
        pooled = np.sum(powered, axis=spatial_axes, keepdims=True)
        return (np.power(pooled, 1.0 / p).astype(x.dtype),)
