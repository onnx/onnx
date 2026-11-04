# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np


def scatter_mean(data, updates):
    """Average original data with (destination coordinate, update value) pairs."""
    if data.dtype.kind in "bUSc":
        raise TypeError(
            f"Scatter mean does not support {data.dtype}; "
            "cast data and updates to an integer or floating-point type."
        )
    output = data.copy()
    integer_mean = np.issubdtype(data.dtype, np.integer)
    # Widen sums so integer overflow and low-precision accumulation do not
    # corrupt the mean before the final conversion to the output dtype.
    sums = data.astype(
        object if integer_mean else np.result_type(data.dtype, np.float32)
    )
    counts = np.ones(data.shape, dtype=np.int64)
    seen = np.zeros(data.shape, dtype=np.bool_)
    for index, update in updates:
        value = (
            (int(update) if np.ndim(update) == 0 else update.astype(object))
            if integer_mean
            else update
        )
        sums[index] += value
        counts[index] += 1
        seen[index] = True
    if integer_mean:
        output[seen] = np.floor_divide(sums[seen], counts[seen]).astype(data.dtype)
    else:
        output[seen] = sums[seen] / counts[seen].astype(sums.dtype)
    return output
