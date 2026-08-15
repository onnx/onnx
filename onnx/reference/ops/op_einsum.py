# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ml_dtypes
import numpy as np

from onnx.reference.op_run import OpRun

# NumPy has no einsum loops for bfloat16: the unoptimized path rejects it outright and
# the optimized path silently returns float32. Contract in float32 and round once at the
# end, which also matches how accelerators evaluate bfloat16 einsums.
_ACCUMULATE_IN_FLOAT32 = (np.dtype(ml_dtypes.bfloat16),)


class Einsum(OpRun):
    def _run(self, *args, equation=None):
        if not isinstance(equation, str):
            raise TypeError(f"equation must be string but is {type(equation)!r}.")
        equation = equation.strip()
        if not equation:
            raise TypeError("equation is empty.")
        dtype = args[0].dtype
        if dtype in _ACCUMULATE_IN_FLOAT32:
            promoted = [arg.astype(np.float32) for arg in args]
            return (
                np.asarray(np.einsum(equation, *promoted, optimize=True)).astype(dtype),
            )
        try:
            return (np.einsum(equation, *args, optimize=True),)
        except TypeError:
            return (np.einsum(equation, *args),)
