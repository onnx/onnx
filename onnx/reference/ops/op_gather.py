# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.op_run import OpRun


class Gather(OpRun):
    def _run(self, x, indices, axis=None):
        if not x.flags["C_CONTIGUOUS"]:
            x = np.ascontiguousarray(x)
        if not indices.flags["C_CONTIGUOUS"]:
            indices = indices.ascontiguousarray()
        try:
            return (np.take(x, indices, axis=axis),)
        except TypeError:
            # distribution x86 requires int32.
            return (np.take(x, indices.astype(int), axis=axis),)
