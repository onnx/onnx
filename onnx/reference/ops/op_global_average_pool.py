# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

from onnx.reference.op_run import OpRun


def _global_average_pool(x: Any) -> Any:
    axis = tuple(range(2, np.ndim(x)))
    y = np.average(x, axis=axis)
    for _ in axis:
        y = np.expand_dims(y, -1)
    return y


class GlobalAveragePool(OpRun):
    def _run(self, x):
        self._get_array_api_namespace(x)
        return (_global_average_pool(x).astype(x.dtype),)
