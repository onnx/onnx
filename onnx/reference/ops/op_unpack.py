# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.op_run import OpRun
from onnx.reference.ops._packing import packing_parameters


class Unpack(OpRun):
    def _run(self, x, count, bits=None, radix=None, group_size=None, axis=-1):
        radix, group_size, width = packing_parameters(bits, radix, group_size)
        if x.dtype not in (np.dtype(np.uint8), np.dtype(np.uint32)) or x.ndim == 0:
            raise ValueError(
                f"X must be a UINT8 or UINT32 tensor of rank at least 1; got dtype={x.dtype}, shape={x.shape}."
            )
        if count.dtype != np.int64 or count.ndim != 0 or count < 0:
            raise ValueError(
                f"count must be a nonnegative scalar int64; "
                f"got dtype={count.dtype}, shape={count.shape}, value={count}."
            )
        n = int(count)
        if not -x.ndim <= axis < x.ndim:
            raise ValueError(f"axis {axis} is out of range for rank {x.ndim}.")
        word_bits = x.dtype.itemsize * 8
        x = np.moveaxis(x, axis, -1)
        groups = (n + group_size - 1) // group_size
        expected_words = (groups * width + word_bits - 1) // word_bits
        if x.shape[-1] != expected_words:
            raise ValueError(
                "Packed axis dimension must equal the word count required by the packing attributes; "
                f"count={n} requires {expected_words} words, got {x.shape[-1]}."
            )
        result = np.empty((*x.shape[:-1], n), dtype=np.uint8)
        if x.size == 0:
            return (np.moveaxis(result, -1, axis),)
        for index in np.ndindex(x.shape[:-1]):
            word = sum(
                int(value) << (i * word_bits) for i, value in enumerate(x[index])
            )
            for g in range(groups):
                code = (word >> (g * width)) & ((1 << width) - 1)
                if code >= radix**group_size:
                    raise ValueError(
                        "Packed group index must be less than radix^group_size."
                    )
                for j in range(min(group_size, n - g * group_size)):
                    result[index][g * group_size + j] = code % radix
                    code //= radix
        return (np.moveaxis(result, -1, axis),)
