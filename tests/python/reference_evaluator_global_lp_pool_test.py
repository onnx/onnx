# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ml_dtypes
import numpy as np
import pytest
from numpy.testing import assert_allclose

from onnx import TensorProto
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_opsetid,
    make_tensor_value_info,
)
from onnx.reference import ReferenceEvaluator


def _evaluate(data, expected, p, elem_type, opset):
    x = make_tensor_value_info("X", elem_type, data.shape)
    y = make_tensor_value_info("Y", elem_type, expected.shape)
    node = make_node("GlobalLpPool", ["X"], ["Y"], p=p)
    graph = make_graph([node], "g", [x], [y])
    model = make_model(graph, opset_imports=[make_opsetid("", opset)])
    return ReferenceEvaluator(model).run(None, {"X": data})[0]


@pytest.mark.parametrize(
    ("data", "expected", "p", "elem_type", "opset"),
    [
        (
            np.array([[[-1, 2, -3], [0, 4, -5]]], dtype=np.float16),
            np.array([[[6], [9]]], dtype=np.float16),
            1,
            TensorProto.FLOAT16,
            22,
        ),
        (
            np.arange(1, 9, dtype=np.float64).reshape(1, 1, 2, 2, 2),
            np.array([[[[[10.90272461]]]]], dtype=np.float64),
            3,
            TensorProto.DOUBLE,
            22,
        ),
        (
            np.array([[[1, 2, 2]]], dtype=ml_dtypes.bfloat16),
            np.array([[[3]]], dtype=ml_dtypes.bfloat16),
            2,
            TensorProto.BFLOAT16,
            22,
        ),
        (
            np.array([[[1, 4, 9]]], dtype=np.float32),
            np.array([[[36]]], dtype=np.float32),
            0.5,
            TensorProto.FLOAT,
            1,
        ),
    ],
)
def test_global_lp_pool(data, expected, p, elem_type, opset):
    got = _evaluate(data, expected, p, elem_type, opset)

    assert got.dtype == data.dtype
    assert_allclose(got.astype(np.float64), expected.astype(np.float64), rtol=1e-6)


def test_global_lp_pool_nan_and_inf():
    data = np.array([[[np.nan, 1, 2], [np.inf, 1, 2]]], dtype=np.float32)
    expected = np.array([[[np.nan], [np.inf]]], dtype=np.float32)

    got = _evaluate(data, expected, 2, TensorProto.FLOAT, 22)

    np.testing.assert_array_equal(got, expected)


def test_global_lp_pool_empty_spatial_dimension():
    data = np.empty((2, 3, 0, 4), dtype=np.float32)
    expected = np.zeros((2, 3, 1, 1), dtype=np.float32)

    got = _evaluate(data, expected, 3, TensorProto.FLOAT, 22)

    np.testing.assert_array_equal(got, expected)
