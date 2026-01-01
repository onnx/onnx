# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

from onnx.reference.op_run import OpRun


def _global_max_pool(x: Any) -> Any:
    spatial_shape = np.ndim(x) - 2
    y = x.max(axis=tuple(range(spatial_shape, spatial_shape + 2)))
    for _ in range(spatial_shape):
        y = np.expand_dims(y, -1)
    return y


class GlobalMaxPool(OpRun):
    def _run(self, x):
        self._get_array_api_namespace(x)
        res = _global_max_pool(x)
        return (res,)
