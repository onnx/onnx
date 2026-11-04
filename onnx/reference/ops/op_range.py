# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ml_dtypes
import numpy as np

from onnx import TensorProto
from onnx.reference.op_run import OpRun

_STASH_TYPE_TO_DTYPE: dict[int, np.dtype] = {
    int(TensorProto.FLOAT): np.dtype(np.float32),
    int(TensorProto.DOUBLE): np.dtype(np.float64),
}

_LOW_PRECISION_DTYPES = (np.dtype(np.float16), np.dtype(ml_dtypes.bfloat16))


class Range(OpRun):
    def _run(self, starts, ends, steps, stash_type=None):  # type: ignore[override]
        dtype = starts.dtype
        end_val = ends.item() if isinstance(ends, np.ndarray) else ends
        step_val = steps.item() if isinstance(steps, np.ndarray) else steps
        if stash_type is not None and dtype in _LOW_PRECISION_DTYPES:
            compute_dtype = _STASH_TYPE_TO_DTYPE.get(int(stash_type))
            if compute_dtype is None:
                raise ValueError(
                    f"Unsupported stash_type {stash_type} for Range; expected FLOAT (1) or DOUBLE (11)"
                )
            return (
                np.arange(
                    starts.astype(compute_dtype).item(),
                    float(end_val),
                    float(step_val),
                    dtype=compute_dtype,
                ).astype(dtype),
            )
        return (np.arange(starts.item(), end_val, step_val).astype(dtype),)
