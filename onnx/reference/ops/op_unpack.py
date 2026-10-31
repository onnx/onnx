# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.op_run import OpRun


class Unpack(OpRun):
    def _run(self, x, count, bits=None):
        if bits is None or not 1 <= bits <= 8:
            raise ValueError(f"bits must be between 1 and 8; got {bits!r}.")
        if x.dtype != np.uint8 or x.ndim == 0:
            raise ValueError(
                f"X must be a UINT8 tensor of rank at least 1; got dtype={x.dtype}, shape={x.shape}."
            )
        if count.dtype != np.int64 or count.ndim != 0 or count < 0:
            raise ValueError(
                f"count must be a nonnegative scalar int64; "
                f"got dtype={count.dtype}, shape={count.shape}, value={count}."
            )
        n = int(count)
        expected_bytes = (n * bits + 7) // 8
        if x.shape[-1] != expected_bytes:
            raise ValueError(
                "Packed last dimension must equal ceil(count * bits / 8); "
                f"count={n}, bits={bits} require {expected_bytes} bytes, got {x.shape[-1]}."
            )
        result = np.empty((*x.shape[:-1], n), dtype=np.uint8)
        if x.size == 0:
            return (result,)
        for i in range(n):
            byte, shift = divmod(i * bits, 8)
            code = x[..., byte].astype(np.uint16) >> shift
            if shift + bits > 8:
                code |= x[..., byte + 1].astype(np.uint16) << (8 - shift)
            result[..., i] = (code & ((1 << bits) - 1)).astype(np.uint8)
        return (result,)
