# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from math import erf

import ml_dtypes
import numpy as np

from onnx.reference.op_run import OpRun
from onnx.reference.ops._op_common_gated import _check_gated_inputs

_erf = np.vectorize(erf, otypes=[np.float64])

# The function body computes these types in float32; see the GeGLU schema.
_LOW_PRECISION_DTYPES = (np.dtype(np.float16), np.dtype(ml_dtypes.bfloat16))


class GeGLU(OpRun):
    def _run(self, a, b, approximate=None):
        _check_gated_inputs("GeGLU", a, b)
        if approximate not in ("none", "tanh"):
            raise ValueError(
                "GeGLU attribute 'approximate' must be 'none' or 'tanh', but got "
                f"{approximate!r}."
            )
        compute_dtype = (
            np.dtype(np.float32) if a.dtype in _LOW_PRECISION_DTYPES else a.dtype
        )
        # Constants are typed so NumPy 1 and NumPy 2 compute in the same precision
        # (NEP 50 changed how np.float64 scalars promote with float32 arrays).
        x = a.astype(compute_dtype)
        half = compute_dtype.type(0.5)
        one = compute_dtype.type(1)
        if approximate == "tanh":
            sqrt_two_over_pi = compute_dtype.type(np.sqrt(2 / np.pi))
            c0 = compute_dtype.type(0.044715)
            gate = half * x * (one + np.tanh(sqrt_two_over_pi * (x + c0 * x * x * x)))
        else:
            sqrt_two = compute_dtype.type(np.sqrt(2))
            gate = half * x * (one + _erf(x / sqrt_two).astype(compute_dtype))
        return ((gate * b.astype(compute_dtype)).astype(a.dtype),)
