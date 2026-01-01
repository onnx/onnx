# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

from onnx.reference.ops._op import OpRunBinaryNumpy


class Div(OpRunBinaryNumpy):
    def __init__(self, onnx_node, run_params):
        def func(x, y):
            if issubclass(x.dtype.type, np.integer):
                assert issubclass(y.dtype.type, np.integer)
                return x // y
            return np.divide(x, y)

        OpRunBinaryNumpy.__init__(self, func, onnx_node, run_params)

    def _run(self, a, b):
        xp = self._get_array_api_namespace(a, b)

        # Check if integer type by converting to numpy for dtype inspection
        # Array API doesn't have a standard way to check integer vs float types
        try:
            a_np = np.asarray(a) if not isinstance(a, Any) else a
            is_integer = issubclass(a_np.dtype.type, np.integer)
        except (AttributeError, TypeError):
            # Fallback: check dtype name
            is_integer = "int" in str(a.dtype).lower()

        # For integer division, use floor_divide; for float, use divide
        if is_integer:
            result = xp.floor_divide(a, b)
        else:
            result = xp.divide(a, b)

        res = (result,)
        if hasattr(res[0], "dtype") and res[0].dtype != a.dtype:
            # Preserve original dtype
            if hasattr(xp, "astype"):
                res = (xp.astype(res[0], a.dtype),)
            else:
                res = (res[0].astype(a.dtype),)
        return self._check_and_fix_outputs(res)
