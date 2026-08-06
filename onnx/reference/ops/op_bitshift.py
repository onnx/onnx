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

    def _run(self, x, y):
        dtype = x.dtype
        bit_width = dtype.itemsize * 8
        left = self.direction == "LEFT"
        # A shift by at least the bit width is defined as a full-width shift. NumPy
        # happens to do the same, but leaves that case out of its documented contract,
        # so the defined result is selected here rather than inherited. A negative Y is
        # undefined by the spec; it lands in the same branch, which keeps this
        # implementation deterministic without the operator promising a value.
        in_range = (y >= 0) & (y < bit_width)
        shifted = (np.left_shift if left else np.right_shift)(
            x, np.where(in_range, y, 0).astype(dtype)
        )
        if left or not np.issubdtype(dtype, np.signedinteger):
            saturated = np.zeros(1, dtype=dtype)
        else:
            # An arithmetic shift by width-1 replicates the sign bit, giving exactly
            # 0 or -1 without constructing -1 in an unsigned dtype.
            saturated = np.right_shift(x, dtype.type(bit_width - 1))
        return (np.where(in_range, shifted, saturated).astype(dtype),)
