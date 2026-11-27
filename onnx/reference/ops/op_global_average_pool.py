# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ml_dtypes
import numpy as np

from onnx.reference.op_run import OpRun


def _global_average_pool(x: np.ndarray) -> np.ndarray:
    axis = tuple(range(2, np.ndim(x)))
    # float16 and bfloat16 are computed in float32: bfloat16 stops accumulating
    # at 256 in np.average and returns wrong means, and float16 overflows the
    # element-count cast for spatial sizes >= 65536.
    xc = x.astype(np.float32) if x.dtype in (np.float16, ml_dtypes.bfloat16) else x
    y = np.average(xc, axis=axis)
    for _ in axis:
        y = np.expand_dims(y, -1)
    return y


class GlobalAveragePool(OpRun):
    def _run(self, x):
        return (_global_average_pool(x).astype(x.dtype),)
