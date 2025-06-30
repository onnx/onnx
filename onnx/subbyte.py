# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import typing

import numpy as np
import numpy.typing as npt
import typing_extensions

if typing.TYPE_CHECKING:
    from collections.abc import Sequence

INT4_MIN = -8
INT4_MAX = 7
UINT4_MIN = 0
UINT4_MAX = 15


@typing_extensions.deprecated(
    "Deprecated since 1.18. Scheduled to remove in 1.20. Consider using libraries like ml_dtypes for dtype conversion",
    category=DeprecationWarning,
)
def float32_to_4bit_unpacked(x: np.ndarray | float, signed: bool) -> np.ndarray:
    """Cast to 4bit via rounding and clipping (without packing).

    Args:
        x: element to be converted
        signed: boolean, whether to convert to signed int4.

    Returns:
        An ndarray with a single int4 element (sign-extended to int8/uint8)
    """
    dtype = np.int8 if signed else np.uint8
    clip_low = INT4_MIN if signed else UINT4_MIN
    clip_high = INT4_MAX if signed else UINT4_MAX
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)

    clipped = np.clip(x, clip_low, clip_high)
    return np.rint(clipped).astype(dtype)  # type: ignore[no-any-return]


@typing_extensions.deprecated(
    "Deprecated since 1.18. Scheduled to remove in 1.20. Consider using libraries like ml_dtypes for dtype conversion",
    category=DeprecationWarning,
)
def float32x2_to_4bitx2(
    val_low: np.dtype, val_high: np.dtype, signed: bool
) -> np.ndarray:
    """Cast two elements to 4bit (via rounding and clipping) and pack
    to a single byte
    Args:
        val_low: element to be packed in the 4 LSB
        val_high: element to be packed in the 4 MSB
        signed: boolean, whether to convert to signed int4.

    Returns:
        An ndarray with a single int8/uint8 element, containing both int4 elements
    """
    i8_high = float32_to_4bit_unpacked(val_high, signed)  # type: ignore[arg-type]
    i8_low = float32_to_4bit_unpacked(val_low, signed)  # type: ignore[arg-type]
    return i8_high << 4 | i8_low & 0x0F


@typing_extensions.deprecated(
    "Deprecated since 1.18. Scheduled to remove in 1.20. Consider using libraries like ml_dtypes for dtype conversion",
    category=DeprecationWarning,
)
def unpack_4bitx2(
    x: npt.NDArray[np.uint8], dims: int | Sequence[int]
) -> npt.NDArray[np.uint8]:
    """Unpack an array of packed uint8 elements (4bitx2) into individual elements
    (still represented as uint8)

    Args:
        x: Input data
        dims: The shape of the output array.

    Returns:
        A array containing unpacked 4-bit elements (as int8/uint8)
    """
    res = np.empty([x.size * 2], dtype=np.uint8)
    x_low = x & np.uint8(0x0F)
    x_high = x & np.uint8(0xF0)
    x_high >>= np.uint8(4)
    res[0::2] = x_low
    res[1::2] = x_high
    if (
        res.size == np.prod(dims) + 1
    ):  # handle single-element padding due to odd number of elements
        res = res.ravel()[:-1]
    res = res.reshape(dims)
    return res


@typing_extensions.deprecated(
    "Deprecated since 1.18. Scheduled to remove in 1.20. Consider using libraries like ml_dtypes for dtype conversion",
    category=DeprecationWarning,
)
def unpack_single_4bitx2(
    x: np.ndarray | np.dtype | float, signed: bool
) -> tuple[np.ndarray, np.ndarray]:
    def unpack_signed(x):
        return np.where((x >> 3) == 0, x, x | 0xF0)

    """Unpack a single byte 4bitx2 to two 4 bit elements
    Args:
        x: Input data
        signed: boolean, whether to interpret as signed int4.
    Returns:
        A tuple of ndarrays containing int4 elements (sign-extended to int8/uint8)
    """
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
    x_low = x & 0x0F
    x_high = x >> 4
    x_low = unpack_signed(x_low) if signed else x_low
    x_high = unpack_signed(x_high) if signed else x_high
    dtype = np.int8 if signed else np.uint8
    return (x_low.astype(dtype), x_high.astype(dtype))


@typing_extensions.deprecated(
    "Deprecated since 1.18. Scheduled to remove in 1.20. Consider using libraries like ml_dtypes for dtype conversion",
    category=DeprecationWarning,
)
def float32_to_float4e2m1_unpacked(values: np.ndarray) -> np.ndarray:
    """Cast float32 to float4e2m1 (without packing).

    Args:
        values: element or array to be converted

    Returns:
        An ndarray with unpacked float4e2m1 elements (as uint8)
    """
    sign = np.where(np.signbit(values), 0x8, 0x0).astype(np.uint8)
    magnitude = np.abs(values)
    res = np.zeros(values.shape, dtype=np.uint8)
    res[(magnitude > 0.25) & (magnitude < 0.75)] = 0x1  # noqa: PLR2004
    res[(magnitude >= 0.75) & (magnitude <= 1.25)] = 0x2  # noqa: PLR2004
    res[(magnitude > 1.25) & (magnitude < 1.75)] = 0x3  # noqa: PLR2004
    res[(magnitude >= 1.75) & (magnitude <= 2.5)] = 0x4  # noqa: PLR2004
    res[(magnitude > 2.5) & (magnitude < 3.5)] = 0x5  # noqa: PLR2004
    res[(magnitude >= 3.5) & (magnitude <= 5.0)] = 0x6  # noqa: PLR2004
    res[magnitude > 5.0] = 0x7  # noqa: PLR2004
    res |= sign
    res[np.isnan(values)] = 0x7
    return res


@typing_extensions.deprecated(
    "Deprecated since 1.18. Scheduled to remove in 1.20. Consider using libraries like ml_dtypes for dtype conversion",
    category=DeprecationWarning,
)
def float32x2_to_float4e2m1x2(val_low: np.ndarray, val_high: np.ndarray) -> np.ndarray:
    """Cast two elements to float4e2m1 and pack to a single byte
    Args:
        val_low: element to be packed in the 4 LSB
        val_high: element to be packed in the 4 MSB

    Returns:
        An ndarray with uint8 elements, containing both float4e2m1 elements
    """
    i8_high = float32_to_float4e2m1_unpacked(val_high)
    i8_low = float32_to_float4e2m1_unpacked(val_low)
    return i8_high << 4 | i8_low & 0x0F
