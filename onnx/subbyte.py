# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

import typing

if typing.TYPE_CHECKING:
    import numpy.typing as npt

INT4_MIN = -8
INT4_MAX = 7
UINT4_MIN = 0
UINT4_MAX = 15


def float32_to_int4(
    x: np.ndarray | np.integer | np.floating,
) -> npt.NDArray[np.uint8] | np.uint8:
    """Cast to 4bit singed integer via rounding and clipping.

    The integer is represented as a 8-bit unsigned integer with the values
    clamped to the range [-8, 7] equivalent signed representations.

    Args:
        x: element to be converted.

    Returns:
        An ndarray with a single int4 element (sign-extended to int8/uint8).
    """
    return np.rint(np.clip(x, INT4_MIN, INT4_MAX)).astype(np.uint8)


def float32_to_uint4(
    x: np.ndarray | np.integer | np.floating,
) -> npt.NDArray[np.uint8] | np.uint8:
    """Cast to 4bit unsigned integer via rounding and clipping.

    The integer is represented as a 8-bit unsigned integer with the values
    clamped to the range [0, 15].

    Args:
        x: element to be converted.

    Returns:
        An ndarray with a single int4 element.
    """
    return np.rint(np.clip(x, UINT4_MIN, UINT4_MAX)).astype(np.uint8)


def float32x2_to_4bitx2(
    val_low: np.ndarray | np.integer | np.floating,
    val_high: np.ndarray | np.integer | np.floating,
    signed: bool,
) -> npt.NDArray[np.uint8] | np.uint8:
    """Cast two elements to 4bit (via rounding and clipping) and pack
    to a single byte

    Args:
        val_low: element to be packed in the 4 LSB
        val_high: element to be packed in the 4 MSB
        signed: boolean, whether to convert to signed int4.

    Returns:
        An ndarray with a single int8/uint8 element, containing both int4 elements
    """
    if signed:
        i8_low = float32_to_int4(val_low)
        i8_high = float32_to_int4(val_high)
    else:
        i8_low = float32_to_uint4(val_low)
        i8_high = float32_to_uint4(val_high)
    i8_high <<= 4
    result = i8_high.astype(np.uint8)
    i8_low &= np.uint8(0x0F)
    result |= i8_low
    return result


def _int4_to_int8(x: npt.NDArray[np.uint8]) -> npt.NDArray[np.int8]:
    """Extend 4-bit signed integer to 8-bit signed integer."""
    return np.where((x >> 3) == 0, x, x | 0xF0).astype(np.int8)


def unpack_4bitx2(
    x: npt.NDArray[np.uint8], signed: bool
) -> tuple[npt.NDArray[np.uint8 | np.int8], npt.NDArray[np.uint8 | np.int8]]:
    """Unpack a single byte 4bitx2 to two 4 bit elements
    Args:
        x: Input data
        signed: boolean, whether to interpret as signed int4.

    Returns:
        A tuple of ndarrays containing int4 elements (sign-extended to int8/uint8)
    """
    x_low = x & np.uint8(0x0F)
    x_high = (x >> 4).astype(np.uint8)
    if signed:
        x_low = _int4_to_int8(x_low)
        x_high = _int4_to_int8(x_high)
    return (x_low, x_high)
