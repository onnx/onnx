# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

MIN_RADIX = 2
MAX_RADIX = 256
MAX_BITS = 8


def packing_parameters(
    bits: int | None, radix: int | None, group_size: int | None
) -> tuple[int, int, int]:
    if bits is not None:
        if radix is not None or group_size is not None:
            raise ValueError(
                "Specify either bits or both radix and group_size, not both."
            )
        if not 1 <= bits <= MAX_BITS:
            raise ValueError(f"bits must be between 1 and 8; got {bits!r}.")
        return 1 << bits, 1, bits
    if radix is None or group_size is None:
        raise ValueError("Specify either bits or both radix and group_size.")
    if not MIN_RADIX <= radix <= MAX_RADIX or group_size < 1:
        raise ValueError("radix must be in [2, 256] and group_size must be positive.")
    limit = 1
    for _ in range(group_size):
        limit *= radix
        if limit > (1 << 63) - 1:
            raise ValueError("radix^group_size must fit in a positive INT64.")
    return radix, group_size, (limit - 1).bit_length()
