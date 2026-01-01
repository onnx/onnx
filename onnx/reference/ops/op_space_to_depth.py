# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.op_run import OpRun


class SpaceToDepth(OpRun):
    def _run(self, data, blocksize=None):
        xp = self._get_array_api_namespace(data)
        if len(data.shape) != 4:
            raise RuntimeError(f"Unexpected shape {data.shape!r}.")
        b, C, H, W = data.shape
        tmpshape = (
            b,
            C,
            H // blocksize,
            blocksize,
            W // blocksize,
            blocksize,
        )
        reshaped = xp.reshape(data, tmpshape)
        transposed = xp.transpose(reshaped, [0, 3, 5, 1, 2, 4])
        finalshape = (
            b,
            C * blocksize * blocksize,
            H // blocksize,
            W // blocksize,
        )
        y = xp.reshape(transposed, finalshape).astype(data.dtype)
        return (y,)