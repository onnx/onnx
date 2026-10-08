# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
# mypy: ignore-errors
from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from onnx import TensorProto, checker, helper
from onnx.reference import ReferenceEvaluator

CASES = []
for dtype in ("int32", "uint32"):
    for opset in (13, 18):
        CASES.extend(
            [
                {
                    "name": f"{dtype}-opset{opset}-vector-default-keepdims",
                    "dtype": dtype,
                    "opset": opset,
                    "data": [3, 4],
                    "axes": [0],
                    "keepdims": None,
                    "expected": [25],
                    "output_shape": [1],
                },
                {
                    "name": f"{dtype}-opset{opset}-matrix-negative-axis-pruned",
                    "dtype": dtype,
                    "opset": opset,
                    "data": [[3, 4], [0, 5]],
                    "axes": [-1],
                    "keepdims": 0,
                    "expected": [25, 25],
                    "output_shape": [2],
                },
            ]
        )
CASES.extend(
    [
        {
            "name": "float32-opset13-all-axes-scalar-control",
            "dtype": "float32",
            "opset": 13,
            "data": [3, 4],
            "axes": None,
            "keepdims": 0,
            "expected": 25,
            "output_shape": [],
        },
        {
            "name": "float32-opset18-explicit-axis-keepdims-control",
            "dtype": "float32",
            "opset": 18,
            "data": [[3, 4], [0, 5]],
            "axes": [1],
            "keepdims": 1,
            "expected": [[25], [25]],
            "output_shape": [2, 1],
        },
        {
            "name": "float64-opset13-negative-axis-control",
            "dtype": "float64",
            "opset": 13,
            "data": [[3, 4], [0, 5]],
            "axes": [-1],
            "keepdims": 0,
            "expected": [25, 25],
            "output_shape": [2],
        },
        {
            "name": "float64-opset18-omitted-axes-default-control",
            "dtype": "float64",
            "opset": 18,
            "data": [3, 4],
            "axes": None,
            "keepdims": None,
            "expected": [25],
            "output_shape": [1],
        },
    ]
)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_reduce_sum_square_preserves_input_dtype(case):
    data = np.array(case["data"], dtype=case["dtype"])
    expected = np.array(case["expected"], dtype=case["dtype"])
    tensor_type = {
        "int32": TensorProto.INT32,
        "uint32": TensorProto.UINT32,
        "float32": TensorProto.FLOAT,
        "float64": TensorProto.DOUBLE,
    }[case["dtype"]]
    attributes = {}
    if case["keepdims"] is not None:
        attributes["keepdims"] = case["keepdims"]
    inputs = ["X"]
    initializers = []
    if case["axes"] is not None:
        if case["opset"] < 18:
            attributes["axes"] = case["axes"]
        else:
            inputs.append("Axes")
            initializers.append(
                helper.make_tensor(
                    "Axes", TensorProto.INT64, [len(case["axes"])], case["axes"]
                )
            )
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("ReduceSumSquare", inputs, ["Y"], **attributes)],
            case["name"],
            [helper.make_tensor_value_info("X", tensor_type, list(data.shape))],
            [helper.make_tensor_value_info("Y", tensor_type, case["output_shape"])],
            initializer=initializers,
        ),
        opset_imports=[helper.make_opsetid("", case["opset"])],
    )
    checker.check_model(model, full_check=True)
    evaluator = ReferenceEvaluator(model)
    (actual,) = evaluator.run(None, {"X": data})
    assert_array_equal(actual, expected)
    assert actual.shape == expected.shape
    assert actual.dtype == data.dtype
