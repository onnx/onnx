# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ml_dtypes
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import onnx
from onnx import defs, helper
from onnx.reference import ReferenceEvaluator


def _get_logsoftmax_function(opset, axis, dtype):
    node = helper.make_node("LogSoftmax", ["x"], ["y"], axis=axis)
    input_type = helper.make_tensor_type_proto(
        helper.np_dtype_to_tensor_dtype(np.dtype(dtype)), None
    )
    schema = defs.get_schema("LogSoftmax", opset)
    function = onnx.FunctionProto.FromString(
        schema.get_context_dependent_function_with_opset_version(
            opset, node.SerializeToString(), [input_type.SerializeToString()]
        )
    )
    onnx.checker.check_function(function)
    output_types = onnx.shape_inference.infer_function_output_types(
        function, [input_type], list(node.attribute)
    )
    assert output_types[0].tensor_type.elem_type == input_type.tensor_type.elem_type
    return function


@pytest.mark.parametrize("opset", [13, 18, 29])
@pytest.mark.parametrize(
    "dtype", [np.float16, ml_dtypes.bfloat16, np.float32, np.float64]
)
def test_logsoftmax_function_opset_dependencies(opset, dtype):
    function = _get_logsoftmax_function(opset, -1, dtype)
    assert defs.get_schema("LogSoftmax", opset).since_version == 13
    nodes = function.node
    assert not any(node.op_type in {"If", "ArgMax"} for node in nodes)
    assert any(node.op_type == "Log1p" for node in nodes) == (opset == 29)
    assert any(o.domain == "" and o.version == opset for o in function.opset_import)


@pytest.mark.parametrize(
    "dtype,margin",
    [
        (np.float16, 10.0),
        (ml_dtypes.bfloat16, 10.0),
        (np.float32, 18.0),
        (np.float64, 40.0),
    ],
)
@pytest.mark.parametrize("axis", [0, 1, -1])
def test_logsoftmax_function_preserves_small_log_probability(dtype, margin, axis):
    x = np.full((3, 3), -margin, dtype=dtype)
    np.fill_diagonal(x, 0)
    x = np.stack([x, x])
    normalizer = np.log1p(2 * np.exp(-margin))
    expected = (x.astype(np.float64) - normalizer).astype(dtype)
    x = np.moveaxis(x, -1, axis)
    expected = np.moveaxis(expected, -1, axis)
    original = x.copy()
    function = _get_logsoftmax_function(29, axis, dtype)

    (actual,) = ReferenceEvaluator(function).run(None, {"input": x}, attributes={})

    assert actual.dtype == dtype
    assert (actual < 0).all()
    assert_array_equal(x, original)
    if dtype == ml_dtypes.bfloat16:
        # NumPy 1.26 cannot promote bfloat16 with the Python float used by isclose.
        actual = actual.astype(np.float32)
        expected = expected.astype(np.float32)
    assert_allclose(
        actual, expected, rtol=1e-2 if dtype == ml_dtypes.bfloat16 else 1e-3, atol=0
    )


@pytest.mark.parametrize(
    "shape,axis",
    [
        ((2, 1, 3), 1),
        ((0, 3), 1),
        ((2, 0, 3), 1),
        ((2, 0, 3), -2),
        ((2, 3, 0), -1),
        ((0, 2, 3), 0),
        ((0,), -1),
        ((0, 0), 0),
        ((2, 0, 0), -1),
    ],
)
@pytest.mark.parametrize(
    "dtype", [np.float16, ml_dtypes.bfloat16, np.float32, np.float64]
)
def test_logsoftmax_function_single_class_and_empty_input(shape, axis, dtype):
    x = np.zeros(shape, dtype=dtype)
    function = _get_logsoftmax_function(29, axis, x.dtype)
    (actual,) = ReferenceEvaluator(function).run(None, {"input": x}, attributes={})
    assert actual.dtype == dtype
    assert_array_equal(actual, x)


@pytest.mark.parametrize(
    "dtype", [np.float16, ml_dtypes.bfloat16, np.float32, np.float64]
)
@pytest.mark.parametrize("axis", [0, 1, -1])
def test_logsoftmax_function_tied_maxima_and_masked_classes(dtype, axis):
    x = np.array([[0, -2, 0], [4, 4, 4], [0, -np.inf, -np.inf]], dtype=dtype)
    normalizer = np.log(2 + np.exp(-2))
    expected = np.array(
        [
            [-normalizer, -2 - normalizer, -normalizer],
            [-np.log(3), -np.log(3), -np.log(3)],
            [0, -np.inf, -np.inf],
        ],
        dtype=dtype,
    )
    x = np.moveaxis(x, -1, axis)
    expected = np.moveaxis(expected, -1, axis)
    original = x.copy()
    function = _get_logsoftmax_function(29, axis, x.dtype)
    (actual,) = ReferenceEvaluator(function).run(None, {"input": x}, attributes={})
    assert actual.dtype == dtype
    assert_array_equal(x, original)
    assert_allclose(
        actual.astype(np.float64),
        expected.astype(np.float64),
        rtol=1e-2 if dtype == ml_dtypes.bfloat16 else 1e-3,
        atol=0,
    )


@pytest.mark.parametrize(
    "dtype", [np.float16, ml_dtypes.bfloat16, np.float32, np.float64]
)
@pytest.mark.parametrize("classes", [264, 300, 2052, 70000])
def test_logsoftmax_function_many_tied_maxima(dtype, classes):
    x = np.zeros((2, classes), dtype=dtype)
    expected = np.full_like(x, -np.log(classes))
    function = _get_logsoftmax_function(29, -1, x.dtype)
    (actual,) = ReferenceEvaluator(function).run(None, {"input": x}, attributes={})
    assert actual.dtype == dtype
    assert_array_equal(actual.astype(np.float64), expected.astype(np.float64))


@pytest.mark.parametrize("dtype", [np.float16, ml_dtypes.bfloat16])
def test_logsoftmax_function_many_nonmaximal_classes(dtype):
    x = np.full((1, 200000), -1, dtype=dtype)
    x[0, 0] = 0
    expected = (x.astype(np.float64) - np.log1p(199999 * np.exp(-1))).astype(dtype)
    function = _get_logsoftmax_function(29, -1, dtype)
    (actual,) = ReferenceEvaluator(function).run(None, {"input": x}, attributes={})
    assert actual.dtype == dtype
    assert_allclose(
        actual.astype(np.float64),
        expected.astype(np.float64),
        rtol=1e-2 if dtype == ml_dtypes.bfloat16 else 1e-3,
        atol=0,
    )


@pytest.mark.parametrize("x", [[np.inf, 0], [-np.inf, -np.inf], [np.nan, 0]])
def test_logsoftmax_function_nonfinite_normalizer(x):
    x = np.array([x], dtype=np.float64)
    function = _get_logsoftmax_function(29, -1, x.dtype)
    with np.errstate(invalid="ignore"):
        (actual,) = ReferenceEvaluator(function).run(None, {"input": x}, attributes={})
    assert np.isnan(actual).all()


@pytest.mark.parametrize(
    "dtype", [np.float16, ml_dtypes.bfloat16, np.float32, np.float64]
)
def test_log1p_preserves_signed_zero_and_input(dtype):
    x = np.array([-0.0, 0.0], dtype=dtype)
    node = helper.make_node("Log1p", ["x"], ["y"])
    (actual,) = ReferenceEvaluator(node).run(None, {"x": x})
    assert actual.dtype == dtype
    assert_array_equal(np.signbit(actual), [True, False])
    assert_array_equal(np.signbit(x), [True, False])


def test_log1p_is_not_available_before_opset29():
    with pytest.raises(onnx.defs.SchemaError):
        defs.get_schema("Log1p", 28)
