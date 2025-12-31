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
                return x // y
            return np.divide(x, y)

        OpRunBinaryNumpy.__init__(self, func, onnx_node, run_params)

    def _run(self, a, b):
        xp = self._get_array_api_namespace(a, b)
        # For integer division, use floor_divide; for float, use divide
        if hasattr(a.dtype, 'type') and issubclass(a.dtype.type, np.integer):
            assert hasattr(b.dtype, 'type') and issubclass(b.dtype.type, np.integer)
            result = xp.floor_divide(a, b)
        else:
            result = xp.divide(a, b)
        
        res = (result,)
        if res[0].dtype != a.dtype:
            # Use astype via the namespace if available
            if hasattr(xp, 'astype'):
                res = (xp.astype(res[0], a.dtype),)
            else:
                res = (res[0].astype(a.dtype),)
        return self._check_and_fix_outputs(res)
