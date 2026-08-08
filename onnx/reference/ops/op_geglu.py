# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from math import erf

import numpy as np

from onnx.reference.op_run import OpRun

_erf = np.vectorize(erf, otypes=[np.float64])


class GeGLU(OpRun):
    def _run(self, a, b, approximate=None):
        # GeGLU requires identical shapes and dtypes for A and B: broadcasting is
        # not applied, matching the equal-shape/no-broadcast contract enforced by
        # GatedActivationShapeInference at graph-build time.
        if a.shape != b.shape:
            raise ValueError(
                "GeGLU requires inputs A and B to have identical shapes "
                f"(broadcasting is not applied), but got A.shape={a.shape} and "
                f"B.shape={b.shape}."
            )
        if a.dtype != b.dtype:
            raise ValueError(
                "GeGLU requires inputs A and B to have identical dtypes, but "
                f"got A.dtype={a.dtype} and B.dtype={b.dtype}."
            )
        approximate = "none" if approximate is None else approximate
        # The gate is exactly the Gelu operator; `approximate` selects the same two
        # formulations Gelu defines, and is forwarded to Gelu by the function body.
        if approximate == "tanh":
            gate = (
                0.5
                * a
                * (1 + np.tanh(np.sqrt(2 / np.pi) * (a + 0.044715 * np.power(a, 3))))
            )
        elif approximate == "none":
            gate = 0.5 * a * (1 + _erf(a / np.sqrt(2)))
        else:
            raise ValueError(
                "GeGLU attribute 'approximate' must be 'none' or 'tanh', but got "
                f"{approximate!r}."
            )
        return (gate.astype(a.dtype) * b,)
