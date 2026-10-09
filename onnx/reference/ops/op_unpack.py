# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.op_run import OpRun


class Unpack(OpRun):
    def _run(self, x, count, bits=None):
        if bits is None or not 1 <= bits <= 8:
            raise ValueError("bits must be between 1 and 8.")
        if x.dtype != np.uint8 or x.ndim == 0:
            raise ValueError("X must be a UINT8 tensor of rank at least 1.")
        if count.dtype != np.int64 or count.ndim != 0 or count < 0:
            raise ValueError("count must be a nonnegative scalar int64.")
        n = int(count)
        if x.shape[-1] != (n * bits + 7) // 8:
            raise ValueError("Packed last dimension must equal ceil(count * bits / 8).")
        result = np.empty((*x.shape[:-1], n), dtype=np.uint8)
        for i in range(n):
            byte, shift = divmod(i * bits, 8)
            code = x[..., byte].astype(np.uint16) >> shift
            if shift + bits > 8:
                code |= x[..., byte + 1].astype(np.uint16) << (8 - shift)
            result[..., i] = (code & ((1 << bits) - 1)).astype(np.uint8)
        return (result,)
