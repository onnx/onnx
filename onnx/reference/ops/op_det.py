# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ml_dtypes
import numpy as np

from onnx.reference.op_run import OpRun


class Det(OpRun):
    def _run(self, x):
        # Compute low-precision inputs in float32, then restore the input dtype.
        data = (
            x.astype(np.float32) if x.dtype in (np.float16, ml_dtypes.bfloat16) else x
        )
        return (np.array(np.linalg.det(data), dtype=x.dtype),)
