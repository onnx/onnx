# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.op_run import OpRun


class Pack(OpRun):
    def _run(self, x, bits=None):
        if bits is None or not 1 <= bits <= 8:
            raise ValueError("bits must be between 1 and 8.")
        if x.dtype != np.uint8 or x.ndim == 0:
            raise ValueError("X must be a UINT8 tensor of rank at least 1.")
        if np.any(x > (1 << bits) - 1):
            raise ValueError("Input codes must be representable in bits bits.")
        shape = (*x.shape[:-1], (x.shape[-1] * bits + 7) // 8)
        result = np.zeros(shape, dtype=np.uint8)
        for i in range(x.shape[-1]):
            byte, shift = divmod(i * bits, 8)
            code = x[..., i].astype(np.uint16)
            result[..., byte] |= ((code << shift) & 255).astype(np.uint8)
            if shift + bits > 8:
                result[..., byte + 1] |= (code >> (8 - shift)).astype(np.uint8)
        return (result,)
