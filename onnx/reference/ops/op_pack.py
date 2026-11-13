# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx import TensorProto
from onnx.reference.op_run import OpRun
from onnx.reference.ops._packing import packing_parameters


class Pack(OpRun):
    def _run(
        self,
        x,
        bits=None,
        radix=None,
        group_size=None,
        axis=-1,
        output_type=TensorProto.UINT8,
    ):
        radix, group_size, width = packing_parameters(bits, radix, group_size)
        if x.dtype != np.uint8 or x.ndim == 0:
            raise ValueError(
                f"X must be a UINT8 tensor of rank at least 1; got dtype={x.dtype}, shape={x.shape}."
            )
        if np.any(x > radix - 1):
            raise ValueError(
                f"Input codes must be representable in radix {radix}; "
                f"expected values in [0, {radix - 1}], got maximum {x.max()}."
            )
        if not -x.ndim <= axis < x.ndim:
            raise ValueError(f"axis {axis} is out of range for rank {x.ndim}.")
        if output_type not in (TensorProto.UINT8, TensorProto.UINT32):
            raise ValueError("output_type must be UINT8 (2) or UINT32 (12).")
        dtype = np.dtype(np.uint8 if output_type == TensorProto.UINT8 else np.uint32)
        word_bits = dtype.itemsize * 8
        x = np.moveaxis(x, axis, -1)
        groups = (x.shape[-1] + group_size - 1) // group_size
        shape = (*x.shape[:-1], (groups * width + word_bits - 1) // word_bits)
        result = np.zeros(shape, dtype=dtype)
        if x.size == 0:
            return (np.moveaxis(result, -1, axis),)
        if group_size == 1:
            code_bits = np.unpackbits(x[..., None], axis=-1, bitorder="little")
            stream = code_bits[..., :width].reshape(*x.shape[:-1], -1)
            packed = np.packbits(stream, axis=-1, bitorder="little")
            byte_count = shape[-1] * dtype.itemsize
            padded = np.zeros((*x.shape[:-1], byte_count), dtype=np.uint8)
            padded[..., : packed.shape[-1]] = packed
            result = padded.view(dtype.newbyteorder("<")).astype(dtype, copy=False)
            return (np.moveaxis(result, -1, axis),)
        for index in np.ndindex(x.shape[:-1]):
            for g in range(groups):
                start = g * group_size
                digits = x[index][start : start + group_size]
                code = sum(int(digit) * radix**j for j, digit in enumerate(digits))
                word_index, shift = divmod(g * width, word_bits)
                remaining = width
                while remaining:
                    take = min(remaining, word_bits - shift)
                    result[index][word_index] = int(result[index][word_index]) | (
                        (code & ((1 << take) - 1)) << shift
                    )
                    code >>= take
                    remaining -= take
                    word_index += 1
                    shift = 0
        return (np.moveaxis(result, -1, axis),)
