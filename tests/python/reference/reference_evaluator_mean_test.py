# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

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
