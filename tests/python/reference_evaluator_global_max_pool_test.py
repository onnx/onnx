# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from onnx.helper import make_node
from onnx.reference import ReferenceEvaluator


@pytest.mark.parametrize(
    ("x", "expected"),
    [
        (
            np.array(
                [
                    [[1, 3, 2, 0], [4, -1, 2, 5], [-2, -3, -1, -4]],
                    [[8, 6, 7, 5], [0, 1, 2, 3], [9, 11, 10, 4]],
                ],
                dtype=np.float32,
            ),
            np.array([[[3], [5], [-1]], [[8], [3], [11]]], dtype=np.float32),
        ),
        (
            np.array(
                [
                    [
                        [[1, 2, 3], [4, 5, 6]],
                        [[9, 8, 7], [6, 5, 4]],
                    ]
                ],
                dtype=np.float64,
            ),
            np.array([6, 9], dtype=np.float64).reshape(1, 2, 1, 1),
        ),
        (
            np.array(
                [
                    [
                        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                        [[[9, 8], [7, 6]], [[5, 4], [3, 2]]],
                    ]
                ],
                dtype=np.float32,
            ),
            np.array([8, 9], dtype=np.float32).reshape(1, 2, 1, 1, 1),
        ),
        (
            np.arange(32, dtype=np.float64).reshape(1, 2, 2, 2, 2, 2),
            np.array([15, 31], dtype=np.float64).reshape(1, 2, 1, 1, 1, 1),
        ),
    ],
    ids=["ncw", "nchw", "ncdhw", "four-spatial-dimensions"],
)
def test_global_max_pool_spatial_ranks(x, expected):
    node = make_node("GlobalMaxPool", ["X"], ["Y"])
    got = ReferenceEvaluator(node).run(None, {"X": x})[0]

    assert got.shape == expected.shape
    assert got.dtype == expected.dtype
    assert_allclose(got, expected)


def test_global_max_pool_nan():
    x = np.array([[[np.nan, 1], [2, 3]]], dtype=np.float64)
    expected = np.array([[[np.nan], [3]]], dtype=np.float64)
    node = make_node("GlobalMaxPool", ["X"], ["Y"])

    got = ReferenceEvaluator(node).run(None, {"X": x})[0]

    assert_allclose(got, expected)


def test_global_max_pool_zero_batch():
    x = np.empty((0, 2, 3), dtype=np.float32)
    node = make_node("GlobalMaxPool", ["X"], ["Y"])

    got = ReferenceEvaluator(node).run(None, {"X": x})[0]

    assert got.shape == (0, 2, 1)
    assert got.dtype == x.dtype


def test_global_max_pool_empty_spatial_dimension():
    x = np.empty((1, 2, 0), dtype=np.float32)
    node = make_node("GlobalMaxPool", ["X"], ["Y"])

    with pytest.raises(ValueError, match="zero-size array"):
        ReferenceEvaluator(node).run(None, {"X": x})
