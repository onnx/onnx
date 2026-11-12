# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ml_dtypes
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import onnx
from onnx import checker, defs, helper, inliner, shape_inference
from onnx.reference import ReferenceEvaluator
from onnx.reference.op_run import OpRunExpand


class Softplus(OpRunExpand):
    pass


class Elu(OpRunExpand):
    pass


class Selu(OpRunExpand):
    pass


class Celu(OpRunExpand):
    pass


class Mish(OpRunExpand):
    pass


class ReduceLogSumExp(OpRunExpand):
    pass


class LogSoftmax(OpRunExpand):
    pass


class SoftmaxCrossEntropyLoss(OpRunExpand):
    pass


_FLOAT_TYPES = [np.float16, ml_dtypes.bfloat16, np.float32, np.float64]
_OPSET = 29
_FUNCTION_OPS = [
    Softplus,
    Elu,
    Selu,
    Celu,
    Mish,
    ReduceLogSumExp,
    SoftmaxCrossEntropyLoss,
    LogSoftmax,
]
_FUNCTIONS = [op.__name__ for op in _FUNCTION_OPS]


def _expand(op, feeds, output_shapes, *, opset=_OPSET, inline=True, **attributes):
    node = helper.make_node(op, list(feeds), list(output_shapes), **attributes)
    dtype = helper.np_dtype_to_tensor_dtype(next(iter(feeds.values())).dtype)
    model = helper.make_model(
        helper.make_graph(
            [node],
            "stable_math",
            [
                helper.make_tensor_value_info(
                    name,
                    helper.np_dtype_to_tensor_dtype(value.dtype),
                    [None] * value.ndim,
                )
                for name, value in feeds.items()
            ],
            [
                helper.make_tensor_value_info(name, dtype, shape)
                for name, shape in output_shapes.items()
            ],
        ),
        opset_imports=[helper.make_opsetid("", opset)],
    )
    if not inline:
        return model, ReferenceEvaluator(model, new_ops=_FUNCTION_OPS)
    expanded = inliner.inline_selected_functions(
        model, [("", name) for name in _FUNCTIONS], inline_schema_functions=True
    )
    assert not any(node.op_type in _FUNCTIONS for node in expanded.graph.node)
    checker.check_model(expanded)
    inferred = shape_inference.infer_shapes(expanded, strict_mode=True)
    checker.check_model(inferred)
    return expanded, ReferenceEvaluator(inferred)


def _op_types(nodes):
    for node in nodes:
        yield node.op_type
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                yield from _op_types(attr.g.node)


def _assert_close(actual, expected):
    assert actual.dtype == expected.dtype
    tolerance = 0.008 if actual.dtype == ml_dtypes.bfloat16 else 0.001
    if actual.dtype == np.float64:
        tolerance = 2e-14
    elif actual.dtype == np.float32:
        tolerance = 2e-6
    assert_allclose(
        actual.astype(np.float64), expected.astype(np.float64), rtol=tolerance, atol=0
    )


@pytest.mark.parametrize(
    "op", ["Softplus", "Elu", "Selu", "Celu", "Mish", "ReduceLogSumExp"]
)
def test_stable_math_function_versions(op):
    schema = defs.get_schema(op, 29)
    legacy_version = max(v for v in schema.function_opset_versions if v < _OPSET)
    old = onnx.FunctionProto.FromString(
        schema.get_function_with_opset_version(legacy_version)
    )
    new = onnx.FunctionProto.FromString(schema.get_function_with_opset_version(29))
    checker.check_function(old)
    checker.check_function(new)
    assert next(o.version for o in old.opset_import if o.domain == "") < _OPSET
    assert next(o.version for o in new.opset_import if o.domain == "") == _OPSET


@pytest.mark.parametrize("dtype", _FLOAT_TYPES)
@pytest.mark.parametrize("shape", [(8,), (2, 4), (0, 3), ()])
def test_softplus_function_stability(dtype, shape):
    values = [-1000, -20, -1, -0.0, 0, 1, 100, 1000]
    x = (
        np.array(values, dtype=dtype).reshape(shape)
        if np.prod(shape) == len(values)
        else np.full(shape, -20, dtype=dtype)
    )
    original = x.copy()
    model, evaluator = _expand("Softplus", {"X": x}, {"Y": x.shape})
    assert any(node.op_type == "Log1p" for node in model.graph.node)
    with np.errstate(over="raise", invalid="raise"):
        (actual,) = evaluator.run(None, {"X": x})
    wide = x.astype(np.float64)
    expected = (np.maximum(wide, 0) + np.log1p(np.exp(-np.abs(wide)))).astype(dtype)
    _assert_close(actual, expected)
    assert_array_equal(x, original)


@pytest.mark.parametrize("dtype", _FLOAT_TYPES)
def test_softplus_function_special_values(dtype):
    x = np.array([-np.inf, np.inf, np.nan, -0.0, 0.0], dtype=dtype)
    _, evaluator = _expand("Softplus", {"X": x}, {"Y": x.shape})
    (actual,) = evaluator.run(None, {"X": x})
    expected = np.array([0, np.inf, np.nan, np.log(2), np.log(2)], dtype=dtype)
    _assert_close(actual, expected)
    assert not np.signbit(actual[0])


@pytest.mark.parametrize("op", ["Elu", "Selu", "Celu", "Mish"])
@pytest.mark.parametrize("dtype", _FLOAT_TYPES)
@pytest.mark.parametrize("inline", [False, True])
def test_activation_functions_transitive_precision(op, dtype, inline):
    x = np.array([-20, -1e-7, -0.0, 0.0, 100], dtype=dtype)
    attributes = {"alpha": 0.5} if op in {"Elu", "Selu", "Celu"} else {}
    if op == "Selu":
        attributes["gamma"] = 1.25
    model, evaluator = _expand(
        op, {"X": x}, {"Y": x.shape}, inline=inline, **attributes
    )
    if inline:
        assert ("Log1p" if op == "Mish" else "Expm1") in _op_types(model.graph.node)
    with np.errstate(over="ignore"):
        (actual,) = evaluator.run(None, {"X": x})
        wide = x.astype(np.float64)
        if op == "Mish":
            expected = wide * np.tanh(np.logaddexp(wide, 0))
        elif op == "Celu":
            expected = np.where(wide < 0, 0.5 * np.expm1(wide / 0.5), wide)
        else:
            expected = np.where(wide < 0, 0.5 * np.expm1(wide), wide)
            if op == "Selu":
                expected *= 1.25
    _assert_close(actual, expected.astype(dtype))
    assert_array_equal(np.signbit(actual[2:4]), np.signbit(x[2:4]))


@pytest.mark.parametrize("op", ["Elu", "Selu"])
@pytest.mark.parametrize("dtype", _FLOAT_TYPES)
@pytest.mark.parametrize(
    "alpha", [0.0, -0.0, -0.5, np.inf, -np.inf, np.nan, 1e-30, 1e30]
)
def test_elu_selu_function_coefficient_edges(op, dtype, alpha):
    x = np.array([-np.inf, -1, -0.0, 0.0, 1, np.inf, np.nan], dtype=dtype)
    attributes = {"alpha": alpha}
    if op == "Selu":
        attributes["gamma"] = -1.25
    _, evaluator = _expand(op, {"X": x}, {"Y": x.shape}, **attributes)
    with np.errstate(over="ignore", invalid="ignore"):
        alpha_cast = np.array(np.float32(alpha), dtype=dtype)
        if op == "Selu":
            # The unfactored expression defines the coefficient corner cases.
            negative = alpha_cast * np.exp(x) - alpha_cast
            expected = np.where(x < 0, negative, x) * np.array(-1.25, dtype=dtype)
        else:
            expected = np.where(x < 0, alpha_cast * np.expm1(x), x)
        (actual,) = evaluator.run(None, {"X": x})
    _assert_close(actual, expected.astype(dtype))
    zeros = (expected == 0) & (actual == 0)
    assert_array_equal(np.signbit(actual[zeros]), np.signbit(expected[zeros]))


@pytest.mark.parametrize("op", ["Elu", "Selu", "Celu"])
@pytest.mark.parametrize("dtype", _FLOAT_TYPES)
@pytest.mark.parametrize("inline", [False, True])
def test_activation_function_default_coefficients(op, dtype, inline):
    x = np.array([-1, -1e-7, 1], dtype=dtype)
    _, evaluator = _expand(op, {"X": x}, {"Y": x.shape}, inline=inline)
    (actual,) = evaluator.run(None, {"X": x})
    wide = x.astype(np.float64)
    alpha = np.array(np.float32(1.6732631921768188), dtype=dtype).astype(np.float64)
    gamma = np.array(np.float32(1.0507010221481323), dtype=dtype).astype(np.float64)
    expected = np.where(wide < 0, np.expm1(wide), wide)
    if op == "Selu":
        expected = np.where(wide < 0, alpha * np.expm1(wide), wide) * gamma
    _assert_close(actual, expected.astype(dtype))


def _logsumexp_oracle(x, axes, keepdims, noop):
    result = x.astype(np.float64)
    if axes is None or len(axes) == 0:
        if noop:
            return x.copy()
        axes = tuple(range(x.ndim))
    normalized = sorted({int(axis) % x.ndim for axis in axes}, reverse=True)
    for axis in normalized:
        with np.errstate(invalid="ignore"):
            result = np.logaddexp.reduce(
                result, axis=axis, keepdims=keepdims, initial=-np.inf
            )
    return np.asarray(result, dtype=x.dtype)


@pytest.mark.parametrize("dtype", _FLOAT_TYPES)
@pytest.mark.parametrize("keepdims", [0, 1])
@pytest.mark.parametrize(
    "values",
    [
        [0, -40],
        [1000, 999],
        [-1000, -1001],
        [0, 0, -40],
        [-np.inf, -np.inf],
        [np.inf, 1000, -np.inf],
        [np.nan, np.inf],
        [0, -np.inf],
    ],
)
def test_reduce_log_sum_exp_function_stability(dtype, keepdims, values):
    x = np.array(values, dtype=dtype)
    expected = _logsumexp_oracle(x, None, keepdims, False)
    model, evaluator = _expand(
        "ReduceLogSumExp", {"data": x}, {"reduced": expected.shape}, keepdims=keepdims
    )
    assert any(node.op_type == "Log1p" for node in model.graph.node)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        (actual,) = evaluator.run(None, {"data": x})
    _assert_close(actual, expected)


@pytest.mark.parametrize("keepdims", [0, 1])
@pytest.mark.parametrize("noop", [0, 1])
@pytest.mark.parametrize("axes", [None, (), (0,), (1,), (-1,), (0, -1)])
@pytest.mark.parametrize("shape", [(2, 3, 4), (0, 3, 4), (2, 0, 4), (2, 3, 0)])
def test_reduce_log_sum_exp_function_dynamic_axes_and_empty(
    shape, axes, noop, keepdims
):
    x = (np.arange(np.prod(shape), dtype=np.float64).reshape(shape) - 12) * 100
    expected = _logsumexp_oracle(x, axes, keepdims, noop)
    feeds = {"data": x}
    if axes is not None:
        feeds["axes"] = np.array(axes, dtype=np.int64)
    _, evaluator = _expand(
        "ReduceLogSumExp",
        feeds,
        {"reduced": expected.shape},
        keepdims=keepdims,
        noop_with_empty_axes=noop,
    )
    (actual,) = evaluator.run(None, feeds)
    _assert_close(actual, expected)


@pytest.mark.parametrize("keepdims", [0, 1])
@pytest.mark.parametrize("noop", [0, 1])
def test_reduce_log_sum_exp_function_scalar(keepdims, noop):
    x = np.array(1000, dtype=np.float64)
    _, evaluator = _expand(
        "ReduceLogSumExp",
        {"data": x},
        {"reduced": ()},
        keepdims=keepdims,
        noop_with_empty_axes=noop,
    )
    (actual,) = evaluator.run(None, {"data": x})
    assert_array_equal(actual, x)


def test_reduce_log_sum_exp_function_unknown_rank():
    schema = defs.get_schema("ReduceLogSumExp", 29)
    function = onnx.FunctionProto.FromString(schema.get_function_with_opset_version(29))
    evaluator = ReferenceEvaluator(function)
    axes = np.array([-1], dtype=np.int64)
    for shape in ((3,), (2, 1, 3), (2, 0, 3), (1, 2, 3, 4)):
        x = np.arange(np.prod(shape), dtype=np.float64).reshape(shape) - 12
        (actual,) = evaluator.run(
            None,
            {"data": x, "axes": axes},
            attributes={"keepdims": 0, "noop_with_empty_axes": 0},
        )
        _assert_close(actual, _logsumexp_oracle(x, axes, False, False))


@pytest.mark.parametrize(
    "dtype,margin",
    [(np.float16, 10), (ml_dtypes.bfloat16, 10), (np.float32, 20), (np.float64, 40)],
)
@pytest.mark.parametrize("inline", [False, True])
def test_cross_entropy_function_transitive_log1p(dtype, margin, inline):
    x = np.array([[0, -margin, -margin]], dtype=dtype)
    labels = np.array([0], dtype=np.int64)
    feeds = {"scores": x, "labels": labels}
    model, evaluator = _expand(
        "SoftmaxCrossEntropyLoss",
        feeds,
        {"loss": (1,), "log_prob": (1, 3)},
        inline=inline,
        reduction="none",
    )
    if inline:
        assert "Log1p" in _op_types(model.graph.node)
    loss, log_prob = evaluator.run(None, feeds)
    normalizer = np.log1p(2 * np.exp(-margin))
    _assert_close(loss, np.array([normalizer], dtype=dtype))
    _assert_close(log_prob, (x.astype(np.float64) - normalizer).astype(dtype))


@pytest.mark.parametrize("op", ["Softplus", "Mish"])
@pytest.mark.parametrize("inline", [False, True])
def test_activation_function_legacy_dependencies(op, inline):
    x = np.array([-20], dtype=np.float32)
    _, evaluator = _expand(op, {"X": x}, {"Y": x.shape}, opset=28, inline=inline)
    (actual,) = evaluator.run(None, {"X": x})
    assert_array_equal(actual, np.zeros_like(x))


def test_context_dependent_function_selects_latest_applicable_builder():
    schema = defs.get_schema("SoftmaxCrossEntropyLoss", 29)
    node = helper.make_node(
        "SoftmaxCrossEntropyLoss", ["X", "labels"], ["loss"], reduction="none"
    )
    types = [
        helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, [1, 3]),
        helper.make_tensor_type_proto(onnx.TensorProto.INT64, [1]),
    ]
    function = onnx.FunctionProto.FromString(
        schema.get_context_dependent_function_with_opset_version(
            29, node.SerializeToString(), [t.SerializeToString() for t in types]
        )
    )
    checker.check_function(function)
    assert any(node.op_type == "LogSoftmax" for node in function.node)
    assert next(o.version for o in function.opset_import if o.domain == "") == _OPSET
    assert not schema.get_context_dependent_function_with_opset_version(
        12, node.SerializeToString(), [t.SerializeToString() for t in types]
    )
