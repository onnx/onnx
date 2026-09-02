# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

from onnx.reference.op_run import OpRun

if TYPE_CHECKING:
    import numpy as np


def _global_max_pool(x: np.ndarray) -> np.ndarray:
    return x.max(axis=tuple(range(2, x.ndim)), keepdims=True)


class GlobalMaxPool(OpRun):
    def _run(self, x):
        res = _global_max_pool(x)
        return (res,)
