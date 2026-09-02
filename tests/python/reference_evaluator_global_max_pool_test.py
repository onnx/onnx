# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from onnx.helper import make_node
from onnx.reference import ReferenceEvaluator


def _run_global_max_pool(x: np.ndarray) -> np.ndarray:
    node = make_node("GlobalMaxPool", ["X"], ["Y"])
    return ReferenceEvaluator(node).run(None, {"X": x})[0]


@pytest.mark.parametrize(
    "shape",
    [(2, 3, 7), (2, 3, 4, 5), (2, 3, 3, 4, 5), (2, 3, 2, 3, 4, 5)],
    ids=["ncw", "nchw", "ncdhw", "four-spatial-dimensions"],
)
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize("layout", ["contiguous", "strided"])
def test_global_max_pool_spatial_ranks(shape, dtype, layout):
    rng = np.random.default_rng(0)
    if layout == "contiguous":
        x = rng.standard_normal(shape).astype(dtype)
    else:
        expanded_shape = (*shape[:-1], shape[-1] * 2)
        x = rng.standard_normal(expanded_shape).astype(dtype)[..., ::2]
        assert not x.flags.c_contiguous
    expected = np.max(x, axis=tuple(range(2, x.ndim)), keepdims=True)

    got = _run_global_max_pool(x)

    assert got.shape == (shape[0], shape[1], *(1,) * (len(shape) - 2))
    assert got.dtype == dtype
    assert_array_equal(got, expected)


def test_global_max_pool_nan():
    x = np.array([[[np.nan, 1], [2, 3]]], dtype=np.float32)
    expected = np.array([[[np.nan], [3]]], dtype=np.float32)

    got = _run_global_max_pool(x)

    assert_array_equal(got, expected)


@pytest.mark.parametrize("shape", [(0, 2, 3), (0, 2, 3, 4), (0, 2, 3, 4, 5)])
def test_global_max_pool_zero_batch(shape):
    x = np.empty(shape, dtype=np.float32)

    got = _run_global_max_pool(x)

    assert got.shape == (0, 2, *(1,) * (len(shape) - 2))
    assert got.dtype == x.dtype


@pytest.mark.parametrize("shape", [(1, 2, 0), (1, 2, 3, 0), (1, 2, 3, 0, 4)])
def test_global_max_pool_empty_spatial_dimension(shape):
    x = np.empty(shape, dtype=np.float32)

    with pytest.raises(ValueError, match="zero-size array"):
        _run_global_max_pool(x)
