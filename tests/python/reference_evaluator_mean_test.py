# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ml_dtypes
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from onnx import checker, helper
from onnx.reference import ReferenceEvaluator


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize("opset", [8, 13])
@pytest.mark.parametrize(
    "values,expected",
    [
        ([2, [[2, 4], [6, 8]]], [[2, 3], [4, 5]]),
        ([[[2, 4], [6, 8]], 2], [[2, 3], [4, 5]]),
        ([[[2], [4]], [[2, 4, 6]]], [[2, 3, 4], [3, 4, 5]]),
        ([[2, 4], [[2], [4]]], [[2, 3], [3, 4]]),
        ([0, [3, 6], [[0], [3]]], [[1, 2], [2, 3]]),
        ([[2, 4], [4, 6]], [3, 5]),
        ([[2, 4]], [2, 4]),
    ],
)
def test_mean_broadcast(dtype, opset, values, expected):
    inputs = [np.array(value, dtype=dtype) for value in values]
    originals = [value.copy() for value in inputs]
    names = [f"X{i}" for i in range(len(inputs))]
    expected = np.array(expected, dtype=dtype)
    tensor_type = helper.np_dtype_to_tensor_dtype(expected.dtype)
    graph = helper.make_graph(
        [helper.make_node("Mean", names, ["Y"])],
        "mean_broadcast",
        [
            helper.make_tensor_value_info(name, tensor_type, value.shape)
            for name, value in zip(names, inputs, strict=True)
        ],
        [helper.make_tensor_value_info("Y", tensor_type, expected.shape)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    checker.check_model(model, full_check=True)
    (result,) = ReferenceEvaluator(model).run(
        None, dict(zip(names, inputs, strict=True))
    )
    assert result.dtype == expected.dtype
    assert result.shape == expected.shape
    assert_allclose(result, expected)
    for value, original in zip(inputs, originals, strict=True):
        assert_array_equal(value, original)


def _run_mean(inputs, opset):
    names = [f"X{i}" for i in range(len(inputs))]
    tensor_type = helper.np_dtype_to_tensor_dtype(inputs[0].dtype)
    graph = helper.make_graph(
        [helper.make_node("Mean", names, ["Y"])],
        "mean",
        [
            helper.make_tensor_value_info(name, tensor_type, value.shape)
            for name, value in zip(names, inputs, strict=True)
        ],
        [helper.make_tensor_value_info("Y", tensor_type, inputs[0].shape)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    checker.check_model(model, full_check=True)
    (result,) = ReferenceEvaluator(model).run(
        None, dict(zip(names, inputs, strict=True))
    )
    return result


@pytest.mark.parametrize(
    "dtype,opset,values,expected",
    [
        # The float16 sum 80000 overflows (max 65504) although the mean fits.
        (np.float16, 8, [40000, 40000], 40000),
        (np.float16, 13, [40000, 40000], 40000),
        # Summing in bfloat16 rounds 256 + 1 back to 256 and gives 85.5.
        # Mean accepts bfloat16 from opset 13.
        (ml_dtypes.bfloat16, 13, [256, 1, 1], 86),
    ],
)
def test_mean_low_precision_accumulates_in_float32(dtype, opset, values, expected):
    inputs = [np.array([value], dtype=dtype) for value in values]
    result = _run_mean(inputs, opset)
    assert result.dtype == dtype
    assert_array_equal(result.astype(np.float64), [expected])


def test_mean_preserves_double_precision():
    value = np.array([1 + 2**-40], dtype=np.float64)
    result = _run_mean([value, value.copy()], 13)
    assert result.dtype == np.float64
    assert_array_equal(result, value)
