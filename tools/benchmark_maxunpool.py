# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark MaxUnpool's indexed assignment against the former scalar loop."""

from __future__ import annotations

import statistics
import timeit

import numpy as np


def _legacy_scatter(values: np.ndarray, indices: np.ndarray, output_size: int) -> np.ndarray:
    result = np.zeros(output_size, dtype=values.dtype)
    for index, value in zip(indices, values, strict=True):
        result[index] = value
    return result


def _indexed_scatter(values: np.ndarray, indices: np.ndarray, output_size: int) -> np.ndarray:
    result = np.zeros(output_size, dtype=values.dtype)
    result[indices] = values
    return result


def _median_seconds(function, repeat: int = 7) -> float:
    return statistics.median(timeit.repeat(function, number=1, repeat=repeat))


def main() -> None:
    values = np.arange(512 * 512, dtype=np.float32)
    indices = np.arange(values.size - 1, -1, -1, dtype=np.int64)
    legacy = _median_seconds(lambda: _legacy_scatter(values, indices, values.size))
    indexed = _median_seconds(lambda: _indexed_scatter(values, indices, values.size))

    print(f"scalar loop: {legacy:.6f} s")
    print(f"indexed assignment: {indexed:.6f} s")
    print(f"measured ratio: {legacy / indexed:.2f}x")


if __name__ == "__main__":
    main()
