# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math

import ml_dtypes
import numpy as np
import pytest

import onnx
from onnx.reference import ReferenceEvaluator

DTYPES = [np.float16, ml_dtypes.bfloat16, np.float32, np.float64]


def evaluate(x):
    dtype = onnx.helper.np_dtype_to_tensor_dtype(x.dtype)
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Expm1", ["x"], ["y"])],
        "expm1",
        [onnx.helper.make_tensor_value_info("x", dtype, x.shape)],
        [onnx.helper.make_tensor_value_info("y", dtype, x.shape)],
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 29)]
    )
    onnx.checker.check_model(model, full_check=True)
    return ReferenceEvaluator(model).run(None, {"x": x})[0]


@pytest.mark.parametrize("dtype", DTYPES)
def test_special_values_and_signed_zero(dtype):
    x = np.array([-0.0, 0.0, -np.inf, np.inf, np.nan], dtype=dtype)
    y = evaluate(x)
    assert y.dtype == x.dtype
    np.testing.assert_array_equal(y.astype(np.float64), [-0.0, 0.0, -1, np.inf, np.nan])
    np.testing.assert_array_equal(np.signbit(y[:2]), [True, False])


@pytest.mark.parametrize(
    "dtype,tiny",
    [
        (np.float16, 2**-14),
        (ml_dtypes.bfloat16, 2**-10),
        (np.float32, 1e-8),
        (np.float64, 1e-20),
    ],
)
def test_small_results(dtype, tiny):
    x = np.array([-tiny, tiny], dtype=dtype)
    expected = np.array([math.expm1(float(v)) for v in x], dtype=dtype)
    y = evaluate(x)
    np.testing.assert_allclose(
        y.astype(np.float64),
        expected.astype(np.float64),
        rtol=1e-2 if dtype == ml_dtypes.bfloat16 else 1e-3,
        atol=0,
    )
    assert np.all(y != 0)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(), (0, 3), (2, 3)])
def test_shapes_and_finite_values(dtype, shape):
    x = np.full(shape, -0.25, dtype=dtype)
    y = evaluate(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype
    expected = np.full(shape, math.expm1(-0.25), dtype=dtype)
    np.testing.assert_allclose(
        y.astype(np.float64),
        expected.astype(np.float64),
        rtol=1e-2 if dtype == ml_dtypes.bfloat16 else 1e-3,
        atol=0,
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_finite_overflow_and_negative_saturation(dtype):
    x = np.array([1000, -1000], dtype=dtype)
    np.testing.assert_array_equal(evaluate(x).astype(np.float64), [np.inf, -1])
