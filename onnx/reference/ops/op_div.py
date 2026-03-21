# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.ops._op import OpRunBinaryNumpy


class Div(OpRunBinaryNumpy):
    def __init__(self, onnx_node, run_params):
        def func(x, y):
            if issubclass(x.dtype.type, np.integer):
                assert issubclass(y.dtype.type, np.integer)
                # Truncate toward zero for integer division (C++-style).
                # Keep integer math to avoid precision loss for large values.
                q = np.floor_divide(x, y)
                r = np.remainder(x, y)
                needs_adjust = (r != 0) & ((x < 0) ^ (y < 0))
                return q + needs_adjust.astype(q.dtype)
            return np.divide(x, y)

        OpRunBinaryNumpy.__init__(self, func, onnx_node, run_params)

    def _run(self, a, b):
        res = OpRunBinaryNumpy._run(self, a, b)
        if res[0].dtype != a.dtype:
            return (res[0].astype(a.dtype),)
        return res
