# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ml_dtypes
import numpy as np

from onnx.reference.op_run import OpRun


class Sum(OpRun):
    def _run(self, *args):
        dtype = args[0].dtype
        if dtype in (np.float16, np.dtype(ml_dtypes.bfloat16)):
            return (sum(a.astype(np.float32) for a in args).astype(dtype),)
        return (sum(args).astype(dtype),)
