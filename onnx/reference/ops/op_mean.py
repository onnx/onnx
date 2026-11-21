# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ml_dtypes
import numpy as np

from onnx.reference.op_run import OpRun


class Mean(OpRun):
    def _run(self, *args):
        dtype = args[0].dtype
        # The running sum can overflow float16 and bfloat16 even when the mean fits.
        acc_dtype = np.float32 if dtype in (np.float16, ml_dtypes.bfloat16) else dtype
        res = args[0].astype(acc_dtype)
        for m in args[1:]:
            res = res + m.astype(acc_dtype, copy=False)
        return ((res / len(args)).astype(dtype),)
