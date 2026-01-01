# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from onnx.reference.op_run import OpRun


class Einsum(OpRun):
    def _run(self, *args, equation=None):
        xp = self._get_array_api_namespace(*args)
        if not isinstance(equation, str):
            raise TypeError(f"equation must be string but is {type(equation)!r}.")
        equation = equation.strip()
        if not equation:
            raise TypeError("equation is empty.")

        # einsum is available in array-api-compat's linalg namespace
        if hasattr(xp, "linalg") and hasattr(xp.linalg, "einsum"):
            try:
                return (xp.linalg.einsum(equation, *args, optimize=True),)
            except TypeError:
                return (xp.linalg.einsum(equation, *args),)
        else:
            # Fallback for backends without einsum
            import numpy as np
            from onnx.reference.array_api_namespace import convert_to_numpy, asarray

            args_np = [convert_to_numpy(arg) for arg in args]
            try:
                result = np.einsum(equation, *args_np, optimize=True)
            except TypeError:
                result = np.einsum(equation, *args_np)
            return (asarray(result, xp=xp),)
