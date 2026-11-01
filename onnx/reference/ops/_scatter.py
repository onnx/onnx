# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np


def scatter_reduce(data, updates, reduction, include_self):
    """Reduce an iterable of (destination coordinate, update value) pairs."""
    if include_self not in (0, 1):
        raise ValueError(f"Invalid scatter include_self={include_self!r}; use 0 or 1.")
    reductions = {
        "none": lambda _x, y: y,
        "add": np.add,
        "mul": np.multiply,
        "min": np.minimum,
        "max": np.maximum,
        "mean": np.add,
    }
    if reduction not in reductions:
        raise ValueError(
            f"Invalid scatter reduction={reduction!r}; "
            "use none, add, mul, min, max, or mean."
        )
    output = data.copy()
    integer_mean = reduction == "mean" and np.issubdtype(data.dtype, np.integer)
    if reduction == "mean":
        if data.dtype.kind in "bUSc":
            raise TypeError(
                f"Scatter mean does not support {data.dtype}; "
                "cast data and updates to an integer or floating-point type."
            )
        # Widen sums so integer overflow and low-precision accumulation do not
        # corrupt the mean before the final conversion to the output dtype.
        sums = data.astype(
            object if integer_mean else np.result_type(data.dtype, np.float32)
        )
        counts = np.full(data.shape, int(include_self), dtype=np.int64)
    seen = np.zeros(data.shape, dtype=np.bool_)
    for index, update in updates:
        if reduction == "mean":
            value = (
                (int(update) if np.ndim(update) == 0 else update.astype(object))
                if integer_mean
                else update
            )
            if not include_self and not np.any(seen[index]):
                sums[index] = value
            else:
                sums[index] += value
            counts[index] += 1
        elif not include_self and not np.any(seen[index]):
            output[index] = update
        else:
            output[index] = reductions[reduction](output[index], update)
        seen[index] = True
    if reduction == "mean":
        if integer_mean:
            output[seen] = np.floor_divide(sums[seen], counts[seen]).astype(data.dtype)
        else:
            output[seen] = sums[seen] / counts[seen].astype(sums.dtype)
    return output
