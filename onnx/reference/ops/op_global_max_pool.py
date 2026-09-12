# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.op_run import OpRun


def _global_max_pool(x: np.ndarray) -> np.ndarray:
    axis = tuple(range(2, np.ndim(x)))
    return x.max(axis=axis, keepdims=True)


class GlobalMaxPool(OpRun):
    def _run(self, x):
        res = _global_max_pool(x)
        return (res,)
