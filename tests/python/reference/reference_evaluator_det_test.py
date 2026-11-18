# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ml_dtypes
import numpy as np
import pytest
from numpy.testing import assert_allclose

from onnx.helper import make_node
from onnx.reference import ReferenceEvaluator


class TestReferenceEvaluatorDet:
    @pytest.mark.parametrize(
        "dtype", [np.float16, ml_dtypes.bfloat16, np.float32, np.float64]
    )
    @pytest.mark.parametrize(
        "data,expected",
        [
            ([[1, 2], [3, 4]], -2),
            ([[1, 2], [2, 4]], 0),
            (
                [
                    [[[1, 2], [3, 4]], [[1, 2], [2, 4]]],
                    [[[1, 2], [2, 1]], [[1, 0], [0, 1]]],
                ],
                [[-2, 0], [-3, 1]],
            ),
            (np.empty((0, 2, 2)), []),
        ],
        ids=["matrix", "singular", "batch", "empty-batch"],
    )
    def test_det(self, dtype, data, expected):
        x = np.array(data, dtype=dtype)
        expected = np.array(expected, dtype=dtype)
        node = make_node("Det", ["X"], ["Y"])

        (got,) = ReferenceEvaluator(node).run(None, {"X": x})

        assert isinstance(got, np.ndarray)
        assert got.dtype == x.dtype
        assert got.shape == expected.shape
        assert_allclose(got.astype(np.float64), expected.astype(np.float64))

    def test_det_preserves_double_precision(self):
        x = np.array([[1, 1], [1, 1 + 2**-40]], dtype=np.float64)
        node = make_node("Det", ["X"], ["Y"])

        (got,) = ReferenceEvaluator(node).run(None, {"X": x})

        assert got.dtype == x.dtype
        assert_allclose(got, 2**-40, rtol=1e-12, atol=0)
