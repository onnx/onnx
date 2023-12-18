# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple, Union

import numpy as np


def float32_to_4bit_unpacked(
    x: Union[np.ndarray, np.dtype, float], signed: bool
) -> int:
    dtype = np.int8 if signed else np.uint8
    clip_low = -8 if signed else 0
    clip_high = 7 if signed else 15
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)

    return np.rint(np.clip(x, clip_low, clip_high)).astype(dtype)  # type: ignore[no-any-return]


def float32x2_to_4bitx2(
    fval_high: np.dtype, fval_low: np.dtype, signed: bool
) -> Union[np.ndarray, np.dtype, int]:
    i8_high = float32_to_4bit_unpacked(fval_high, signed)
    i8_low = float32_to_4bit_unpacked(fval_low, signed)
    return i8_high << 4 | i8_low & 0x0F


def unpack_single_4bitx2(
    x: Union[np.ndarray, np.dtype, float], signed: bool
) -> Tuple[np.ndarray, np.ndarray]:
    unpack_signed = lambda x: np.where((x >> 3) == 0, x, x | 0xF0)

    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
    x_low = x & 0x0F
    x_high = x >> 4
    x_low = unpack_signed(x_low) if signed else x_low
    x_high = unpack_signed(x_high) if signed else x_high
    dtype = np.int8 if signed else np.uint8
    return (x_high.astype(dtype), x_low.astype(dtype))
