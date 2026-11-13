# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
import pytest

import onnx
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator

_DTYPES = [
    (np.uint8, np.uint8),
    (np.uint8, np.int8),
    (np.int8, np.uint8),
    (np.int8, np.int8),
]


def _check_convolution(x, w, x_zero_point, w_zero_point, expected, **attributes):
    feeds = {"x": x, "w": w}
    inputs = ["x", "w"]
    for name, value in [("x_zero_point", x_zero_point), ("w_zero_point", w_zero_point)]:
        inputs.append(name if value is not None else "")
        if value is not None:
            feeds[name] = value
    node = helper.make_node("ConvInteger", inputs, ["y"], **attributes)
    graph = helper.make_graph(
        [node],
        "conv_integer_zero_points",
        [
            helper.make_tensor_value_info(
                name,
                TensorProto.INT8 if value.dtype == np.int8 else TensorProto.UINT8,
                value.shape,
            )
            for name, value in feeds.items()
        ],
        [helper.make_tensor_value_info("y", TensorProto.INT32, expected.shape)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 10)])
    onnx.checker.check_model(model, full_check=True)
    actual = ReferenceEvaluator(model).run(None, feeds)[0]
    assert actual.dtype == np.int32
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("rank", [1, 2, 3, 4])
@pytest.mark.parametrize(("x_dtype", "w_dtype"), _DTYPES)
@pytest.mark.parametrize("group", [1, 2])
def test_conv_integer_channel_zero_points(rank, x_dtype, w_dtype, group):
    # For rank 3 and group 1, C == M means a misplaced zero-point axis can
    # broadcast successfully while silently changing the numerical result.
    x_shift = 7 if x_dtype == np.int8 else 0
    w_shift = 8 if w_dtype == np.int8 else 0
    x = (np.array([[1, 4], [2, 5]]) - x_shift).astype(x_dtype)
    x = x.reshape((1, 2, 2) + (1,) * (rank - 1))
    x_zero_point = np.array(3 - x_shift, dtype=x_dtype)
    w_zero_point = np.array([2, 5]) - w_shift
    w_zero_point = w_zero_point.astype(w_dtype)
    if group == 1:
        w = np.array([[1, 4], [6, 2]]) - w_shift
        expected = np.array([[0, 3], [1, -5]], dtype=np.int32)
    else:
        w = np.array([[1], [6]]) - w_shift
        expected = np.array([[2, -1], [-1, 2]], dtype=np.int32)
    w = w.astype(w_dtype).reshape((2, 2 // group) + (1,) * rank)
    expected = expected.reshape((1, 2, 2) + (1,) * (rank - 1))
    _check_convolution(x, w, x_zero_point, w_zero_point, expected, group=group)


@pytest.mark.parametrize("rank", [1, 2, 3, 4])
@pytest.mark.parametrize(("x_dtype", "w_dtype"), _DTYPES)
def test_conv_integer_single_channel_vector_zero_point(rank, x_dtype, w_dtype):
    x = np.array([1, 4], dtype=x_dtype).reshape((1, 1, 2) + (1,) * (rank - 1))
    w = np.array([3], dtype=w_dtype).reshape((1, 1) + (1,) * rank)
    expected = np.array([-2, 1], dtype=np.int32).reshape(x.shape)
    _check_convolution(
        x,
        w,
        np.array(3, dtype=x_dtype),
        np.array([2], dtype=w_dtype),
        expected,
    )


@pytest.mark.parametrize("rank", [1, 2, 3, 4])
@pytest.mark.parametrize(("x_dtype", "w_dtype"), _DTYPES)
@pytest.mark.parametrize(
    "mode", ["scalar", "omitted_weight", "omitted_input", "omitted_both"]
)
def test_conv_integer_scalar_and_omitted_zero_points(rank, x_dtype, w_dtype, mode):
    x = np.array([[1, 4], [2, 5]], dtype=x_dtype)
    x = x.reshape((1, 2, 2) + (1,) * (rank - 1))
    w = np.array([[1, 4], [6, 2]], dtype=w_dtype).reshape((2, 2) + (1,) * rank)
    x_zero_point = np.array(3, dtype=x_dtype)
    w_zero_point = np.array(2, dtype=w_dtype)
    if mode == "scalar":
        expected = [[0, 3], [-8, 4]]
    elif mode == "omitted_weight":
        w_zero_point = None
        expected = [[-6, 9], [-14, 10]]
    elif mode == "omitted_input":
        x_zero_point = None
        w_zero_point = np.array([2, 5], dtype=w_dtype)
        expected = [[3, 6], [-5, -11]]
    else:
        x_zero_point = None
        w_zero_point = None
        expected = [[9, 24], [10, 34]]
    expected = np.array(expected, dtype=np.int32).reshape(x.shape)
    _check_convolution(x, w, x_zero_point, w_zero_point, expected)


@pytest.mark.parametrize("rank", [1, 2, 3, 4])
@pytest.mark.parametrize(("x_dtype", "w_dtype"), _DTYPES)
@pytest.mark.parametrize("padding", ["explicit", "SAME_UPPER", "SAME_LOWER"])
def test_conv_integer_channel_zero_points_with_geometry(
    rank, x_dtype, w_dtype, padding
):
    x = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=x_dtype)
    x = x.reshape((1, 2, 5) + (1,) * (rank - 1))
    w = np.array([[[2, 3], [4, 5]], [[5, 7], [6, 4]]], dtype=w_dtype)
    w = w.reshape((2, 2, 2) + (1,) * (rank - 1))
    attributes = {"strides": [2] + [1] * (rank - 1)}
    if padding == "explicit":
        attributes["pads"] = ([1] + [0] * (rank - 1)) * 2
        attributes["dilations"] = [2] + [1] * (rank - 1)
        expected = [[11, 27, 12], [-6, 0, 6]]
    elif padding == "SAME_UPPER":
        attributes["auto_pad"] = padding
        expected = [[17, 29, 14], [-3, 1, 7]]
    else:
        attributes["auto_pad"] = padding
        expected = [[7, 23, 35], [-7, -1, 3]]
    expected = np.array(expected, dtype=np.int32).reshape((1, 2, 3) + (1,) * (rank - 1))
    _check_convolution(
        x,
        w,
        np.array(3, dtype=x_dtype),
        np.array([2, 5], dtype=w_dtype),
        expected,
        **attributes,
    )
