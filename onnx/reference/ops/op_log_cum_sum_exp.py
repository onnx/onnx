# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.op_run import OpRun


class LogCumSumExp(OpRun):
    def _run(self, x, axis=0):
        if x.ndim == 0:
            raise ValueError(
                "LogCumSumExp requires rank >= 1; reshape scalar X to a one-element tensor."
            )
        if not -x.ndim <= axis < x.ndim:
            raise ValueError(
                f"LogCumSumExp axis {axis} is outside [{-x.ndim}, {x.ndim - 1}] "
                f"for X with shape {x.shape}; choose an axis in that range."
            )
        accumulator_dtype = np.float64 if x.dtype == np.float64 else np.float32
        # logaddexp handles both equal infinities without infinity-minus-infinity.
        with np.errstate(invalid="ignore"):
            result = np.logaddexp.accumulate(
                x.astype(accumulator_dtype), axis=axis, dtype=accumulator_dtype
            )
        return (result.astype(x.dtype),)
