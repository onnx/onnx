# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.op_run import OpRun


class QLinearMatMul(OpRun):
    def _run(
        self, a, a_scale, a_zero_point, b, b_scale, b_zero_point, y_scale, y_zero_point
    ):
        A = a.astype(np.int32)
        if a_zero_point is not None:
            zero_point = a_zero_point.astype(np.int32)
            if a.ndim == 2 and zero_point.ndim == 1:
                zero_point = zero_point[:, np.newaxis]
            A -= zero_point
        B = b.astype(np.int32)
        if b_zero_point is not None:
            B -= b_zero_point.astype(np.int32)
        C = np.matmul(A, B)
        if a.ndim == 2 and a_scale.ndim == 1 and b.ndim > 1:
            # A vector quantizes rows of a, not its contraction dimension.
            # A vector b removes the final output axis, so keep that scale 1D.
            a_scale = a_scale[:, np.newaxis]
        # Spec: y = saturate(round(x / y_scale) + y_zero_point), with
        # ties-to-even rounding applied to (x / y_scale) before the zero
        # point is added. Adding first is not equivalent when the quotient
        # is a half-integer and y_zero_point is odd.
        D = np.rint(C * (a_scale * b_scale / y_scale))
        if y_zero_point is not None:
            D += y_zero_point
            dtype = y_zero_point.dtype
        else:
            dtype = a.dtype
        qmin, qmax = np.iinfo(dtype).min, np.iinfo(dtype).max
        return (np.clip(D, qmin, qmax).astype(dtype),)
