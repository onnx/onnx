# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.op_run import OpRun


class Shape_1(OpRun):
    def _run(self, data):
        xp = self._get_array_api_namespace(data)
        # Return shape as array using the same namespace as input
        shape_tuple = data.shape
        # Create array with int64 dtype using array API
        return (xp.asarray(shape_tuple, dtype=xp.int64),)


class Shape_15(Shape_1):
    @staticmethod
    def _interval(n: int, start: int | None, end: int | None) -> tuple[int, int] | None:
        if start == 0:
            if end is None or np.isnan(end):
                return None
            if end < 0:
                return (0, n + end)
            return (0, end)
        if end is None or np.isnan(end):
            return (start, n)
        if end < 0:
            return (start, n + end)
        return (start, end)

    def _run(self, data, end=None, start=None):
        xp = self._get_array_api_namespace(data)
        ab = self._interval(len(data.shape), start=start, end=end)
        if ab is None:
            return (xp.asarray(data.shape, dtype=xp.int64),)
        return (xp.asarray(data.shape[ab[0] : ab[1]], dtype=xp.int64),)
