# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.ops.op_reduce_mean import _mean


def scatter_mean(data, update_pairs):
    """Average data with pairs from a replayable update iterator factory."""
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
    for index, update in update_pairs():
        value = (
            (int(update) if np.ndim(update) == 0 else update.astype(object))
            if integer_mean
            else update
        )
        with np.errstate(over="ignore", invalid="ignore"):
            sums[index] += value
        counts[index] += 1
        seen[index] = True
    if integer_mean:
        output[seen] = np.floor_divide(sums[seen], counts[seen]).astype(data.dtype)
    else:
        output[seen] = sums[seen] / counts[seen].astype(sums.dtype)
        failed = np.flatnonzero(seen & ~np.isfinite(output))
        if failed.size:
            # Replay only failed destinations through ReduceMean's finite-sum
            # overflow fallback, including each original data value once.
            contributions = {i: [data.flat[i]] for i in failed}
            positions = np.arange(data.size).reshape(data.shape)
            for index, update in update_pairs():
                for position, value in zip(
                    np.asarray(positions[index]).flat,
                    np.asarray(update).flat,
                    strict=True,
                ):
                    if position in contributions:
                        contributions[position].append(value)
            for position, values in contributions.items():
                output.flat[position] = _mean(
                    np.asarray(values, dtype=sums.dtype), axes=None, keepdims=False
                )
    return output
