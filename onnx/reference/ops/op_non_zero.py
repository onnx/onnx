# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.op_run import OpRun


class NonZero(OpRun):
    def _run(self, x):
        if x.ndim == 0:
            return (np.empty((0, np.count_nonzero(x)), dtype=np.int64),)
        # Specify np.int64 for Windows x86 machines
        res = np.vstack(np.nonzero(x)).astype(np.int64)
        return (res,)
