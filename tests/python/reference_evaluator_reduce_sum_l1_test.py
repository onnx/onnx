# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ml_dtypes
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from onnx import TensorProto, checker, helper
from onnx.reference import ReferenceEvaluator

_LOW_PRECISION = [np.float16, ml_dtypes.bfloat16]


def _run_reduce(op_type, x, axes, opset, keepdims=1):
    tensor_type = helper.np_dtype_to_tensor_dtype(x.dtype)
    inputs = [helper.make_tensor_value_info("X", tensor_type, x.shape)]
    feeds = {"X": x}
    node_inputs = ["X"]
    attrs = {"keepdims": keepdims}
    if (op_type == "ReduceSum" and opset >= 13) or (
        op_type == "ReduceL1" and opset >= 18
    ):
        axes_arr = np.array(axes, dtype=np.int64)
        inputs.append(
            helper.make_tensor_value_info("axes", TensorProto.INT64, axes_arr.shape)
        )
        feeds["axes"] = axes_arr
        node_inputs.append("axes")
    else:
        attrs["axes"] = list(axes)

    out_shape = [
        1 if i in axes else d
        for i, d in enumerate(x.shape)
        if keepdims or i not in axes
    ]
    node = helper.make_node(op_type, node_inputs, ["Y"], **attrs)
    graph = helper.make_graph(
        [node],
        op_type.lower(),
        inputs,
        [helper.make_tensor_value_info("Y", tensor_type, out_shape)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    checker.check_model(model)
    (result,) = ReferenceEvaluator(model).run(None, feeds)
    return result


@pytest.mark.parametrize("opset", [11, 13])
@pytest.mark.parametrize("dtype", _LOW_PRECISION)
def test_reduce_sum_low_precision_long_slice(opset, dtype):
    # Summing 1024 ones in bfloat16 stalled at 256 when accumulated in the
    # input dtype.
    x = np.ones((2, 1024), dtype=dtype)
    got = _run_reduce("ReduceSum", x, [1], opset=opset)

    assert got.dtype == np.dtype(dtype)
    assert_array_equal(got, np.full((2, 1), 1024, dtype=dtype))


@pytest.mark.parametrize("opset", [11, 13])
def test_reduce_sum_float16_cancelling_partial_sums(opset):
    # Intermediate partial sum exceeds float16 max (65504), while the final
    # sum (40000) is representable.
    x = np.array([[40000.0, 40000.0, -40000.0]], dtype=np.float16)
    got = _run_reduce("ReduceSum", x, [1], opset=opset)

    assert got.dtype == np.float16
    assert_array_equal(got, np.array([[40000.0]], dtype=np.float16))


@pytest.mark.parametrize("opset", [13, 18])
@pytest.mark.parametrize("dtype", _LOW_PRECISION)
def test_reduce_l1_low_precision_long_slice(opset, dtype):
    # Summing 1024 +/-1 values in bfloat16 stalled at 256 when accumulated in
    # the input dtype.
    x = np.ones((2, 1024), dtype=dtype)
    x[:, ::2] = -x[:, ::2]
    got = _run_reduce("ReduceL1", x, [1], opset=opset)

    assert got.dtype == np.dtype(dtype)
    assert_array_equal(got, np.full((2, 1), 1024, dtype=dtype))


@pytest.mark.parametrize("op_type,opset", [("ReduceSum", 13), ("ReduceL1", 18)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64])
def test_reduce_sum_l1_other_dtypes_unchanged(op_type, opset, dtype):
    rng = np.random.default_rng(0)
    if np.issubdtype(dtype, np.integer):
        x = rng.integers(-10, 10, size=(3, 5)).astype(dtype)
    else:
        x = rng.standard_normal((3, 5)).astype(dtype)

    got = _run_reduce(op_type, x, [1], opset=opset)
    expected = (
        np.sum(np.abs(x), axis=1, keepdims=True)
        if op_type == "ReduceL1"
        else np.sum(x, axis=1, keepdims=True)
    ).astype(dtype)

    assert got.dtype == np.dtype(dtype)
    if np.issubdtype(dtype, np.integer):
        assert_array_equal(got, expected)
    else:
        assert_allclose(got, expected, rtol=1e-12 if dtype == np.float64 else 1e-6)
