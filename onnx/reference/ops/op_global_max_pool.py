# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.op_run import OpRun


def _global_max_pool(x: np.ndarray) -> np.ndarray:
    axis = tuple(range(2, np.ndim(x)))
    y = x.max(axis=axis)
    for _ in axis:
        y = np.expand_dims(y, -1)
    return y


class GlobalMaxPool(OpRun):
    def _run(self, x):
        res = _global_max_pool(x)
        return (res,)
