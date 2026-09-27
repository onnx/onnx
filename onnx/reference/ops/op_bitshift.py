# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.ops._op import OpRunBinaryNum


class BitShift(OpRunBinaryNum):
    def __init__(self, onnx_node, run_params):
        OpRunBinaryNum.__init__(self, onnx_node, run_params)
        if self.direction not in ("LEFT", "RIGHT"):
            raise ValueError(f"Unexpected value for direction ({self.direction!r}).")

    def _run(self, a, b):
        a, b = np.broadcast_arrays(a, b)
        bit_width = np.iinfo(a.dtype).bits
        out_of_range = (b < 0) | (b >= bit_width)
        valid = ~out_of_range
        result = np.empty(a.shape, dtype=a.dtype)

        if self.direction == "RIGHT":
            if np.issubdtype(a.dtype, np.signedinteger):
                result[out_of_range] = np.where(a[out_of_range] < 0, -1, 0)
            else:
                result[out_of_range] = 0
            result[valid] = np.right_shift(a[valid], b[valid])
        else:
            result[out_of_range] = 0
            if np.issubdtype(a.dtype, np.signedinteger):
                unsigned_dtype = np.dtype(f"u{a.dtype.itemsize}")
                shifted = np.left_shift(
                    a[valid].view(unsigned_dtype),
                    b[valid].astype(unsigned_dtype, copy=False),
                )
                result[valid] = shifted.view(a.dtype)
            else:
                result[valid] = np.left_shift(a[valid], b[valid])

        return (result,)
