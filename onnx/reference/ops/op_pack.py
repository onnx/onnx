# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.op_run import OpRun
from onnx.reference.ops._packing import packing_parameters


class Pack(OpRun):
    def _run(self, x, bits=None, radix=None, group_size=None):
        radix, group_size, width = packing_parameters(bits, radix, group_size)
        if x.dtype != np.uint8 or x.ndim == 0:
            raise ValueError(
                f"X must be a UINT8 tensor of rank at least 1; got dtype={x.dtype}, shape={x.shape}."
            )
        if np.any(x >= radix):
            raise ValueError(
                f"Input codes must be representable in radix {radix}; "
                f"expected values in [0, {radix - 1}], got maximum {x.max()}."
            )
        groups = (x.shape[-1] + group_size - 1) // group_size
        shape = (*x.shape[:-1], (groups * width + 7) // 8)
        result = np.zeros(shape, dtype=np.uint8)
        if x.size == 0:
            return (result,)
        for index in np.ndindex(x.shape[:-1]):
            word = 0
            for g in range(groups):
                start = g * group_size
                digits = x[index][start : start + group_size]
                code = sum(int(digit) * radix**j for j, digit in enumerate(digits))
                word |= code << (g * width)
            result[index] = np.frombuffer(
                word.to_bytes(shape[-1], "little"), dtype=np.uint8
            )
        return (result,)
