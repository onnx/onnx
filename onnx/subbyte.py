# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import typing
from typing import Sequence

import numpy as np

if typing.TYPE_CHECKING:
    import numpy.typing as npt

INT4_MIN = -8
INT4_MAX = 7
UINT4_MIN = 0
UINT4_MAX = 15


def cast_int4(
    x: np.ndarray | np.integer | np.floating,
) -> npt.NDArray[np.int8] | np.int8:
    """Cast to 4bit singed integer via rounding and clipping.

    The integer is represented as a 8-bit integer with the values
    clamped to the range [-8, 7].

    Args:
        x: element to be converted.

    Returns:
        An ndarray with a single int4 element (sign-extended to int8/uint8).
    """
    return np.rint(np.clip(x, INT4_MIN, INT4_MAX)).astype(np.int8)


def cast_uint4(
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


def pack_4bit(data: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """Convert an unpacked uint4/int4 array to flatten, packed int4/uint4 array."""
    if data.dtype not in (np.int8, np.uint8):
        raise TypeError(
            "Input array must be of type int8/uint8. "
            "Call cast_int4 or cast_uint4 to cast the array first."
        )
    # Create a 1D copy
    array_flat = data.ravel().astype(np.uint8)
    size = data.size
    odd_sized = size % 2 == 1
    if odd_sized:
        array_flat.resize([size + 1], refcheck=False)
    array_flat &= 0x0F
    array_flat[1::2] <<= np.uint8(4)
    return array_flat[0::2] | array_flat[1::2]


def _int4_to_int8(x: npt.NDArray[np.uint8]) -> npt.NDArray[np.int8]:
    """Extend 4-bit signed integer to 8-bit signed integer."""
    return np.where((x >> 3) == 0, x, x | 0xF0).astype(np.int8)


def unpack_int4(
    data: npt.NDArray[np.uint8], dims: Sequence[int], signed: bool
) -> npt.NDArray[np.uint8]:
    """Convert a packed int4/uint4 array to unpacked int4/uint4 array represented as int8/uint8.

    See :ref:`onnx-detail-int4` for technical details.

    Args:
        data: A numpy array.
        dims: The dimensions are used to reshape the unpacked buffer.
        signed: Whether the 4 bit integer is signed or unsigned.

    Returns:
        A numpy array of int8/uint8 reshaped to dims.
    """
    if data.dtype != np.uint8:
        raise TypeError("Input array must be of type int8/uint8.")
    result = np.empty([data.size * 2], dtype=data.dtype)
    array_low = data & np.uint8(0x0F)
    array_high = data & np.uint8(0xF0)
    array_high >>= np.uint8(4)
    result[0::2] = array_low
    result[1::2] = array_high
    if signed:
        result = _int4_to_int8(result)  # type: ignore
    if result.size == np.prod(dims) + 1:
        # handle single-element padding due to odd number of elements
        result = result[:-1]
    result.resize(dims, refcheck=False)
    return result  # type: ignore


def pack_4bit_pair(
    val_low: np.ndarray | np.integer | np.floating,
    val_high: np.ndarray | np.integer | np.floating,
    signed: bool,
) -> npt.NDArray[np.uint8] | np.uint8:
    """Cast two elements to 4bit (via rounding and clipping) and pack
    to a single byte.

    Args:
        val_low: element to be packed in the 4 LSB
        val_high: element to be packed in the 4 MSB
        signed: boolean, whether to convert to signed int4.

    Returns:
        An ndarray with a single int8/uint8 element, containing both int4 elements
    """
    if signed:
        i8_low = cast_int4(val_low).view(np.uint8)
        i8_high = cast_int4(val_high).view(np.uint8)
    else:
        i8_low = cast_uint4(val_low)
        i8_high = cast_uint4(val_high)
    i8_high <<= 4
    result = i8_high.astype(np.uint8)
    i8_low &= np.uint8(0x0F)
    result |= i8_low
    return result


def unpack_4bit_pair(
    x: npt.NDArray[np.uint8], signed: bool
) -> tuple[npt.NDArray[np.uint8 | np.int8], npt.NDArray[np.uint8 | np.int8]]:
    """Unpack a single byte 4bitx2 to two 4 bit elements.

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
