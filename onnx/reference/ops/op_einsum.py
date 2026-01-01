# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.array_api_namespace import asarray, convert_to_numpy
from onnx.reference.op_run import OpRun


class Einsum(OpRun):
    def _run(self, *args, equation=None):
        xp = self._get_array_api_namespace(*args)
        if not isinstance(equation, str):
            raise TypeError(f"equation must be string but is {type(equation)!r}.")
        equation = equation.strip()
        if not equation:
            raise TypeError("equation is empty.")
        
        # Convert to numpy for einsum (not in array API standard)
        args_np = [convert_to_numpy(arg) for arg in args]
        try:
            result = np.einsum(equation, *args_np, optimize=True)
        except TypeError:
            result = np.einsum(equation, *args_np)
        
        # Convert back to original array type
        return (asarray(result, xp=xp),)
