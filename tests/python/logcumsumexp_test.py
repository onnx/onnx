# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from decimal import Decimal, localcontext

import ml_dtypes
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from onnx import (
    FunctionProto,
    TensorProto,
    checker,
    defs,
    helper,
    inliner,
    shape_inference,
)
from onnx.reference import ReferenceEvaluator
from onnx.reference.op_run import OpRunExpand


class LogCumSumExp(OpRunExpand):
    pass


def test_logcumsumexp_differentiation_category() -> None:
    schema = defs.get_schema("LogCumSumExp", 29)
    differentiable = defs.OpSchema.DifferentiationCategory.Differentiable
    assert schema.inputs[0].differentiation_category == differentiable
    assert schema.outputs[0].differentiation_category == differentiable


def _oracle(x: np.ndarray, axis: int) -> np.ndarray:
    """Sum exponentials independently for each slice in decimal arithmetic."""
    moved = np.moveaxis(x.astype(np.float64), axis, 0)
    expected = np.empty_like(moved)
    with localcontext() as context:
        context.prec = 80
        for index in np.ndindex(moved.shape[1:]):
            total = Decimal(0)
            for j in range(moved.shape[0]):
                value = Decimal(float(moved[(j, *index)]))
                if value.is_nan() or total.is_nan():
                    total = Decimal("NaN")
                else:
                    total += value.exp()
                expected[(j, *index)] = float(total.ln())
    return np.moveaxis(expected, 0, axis).astype(x.dtype)


def _model(x: np.ndarray, axis: int | None, *, unknown_rank: bool = False):
    elem_type = helper.np_dtype_to_tensor_dtype(x.dtype)
    node = helper.make_node(
        "LogCumSumExp", ["X"], ["Y"], **({} if axis is None else {"axis": axis})
    )
    shape = None if unknown_rank else [None] * x.ndim
    graph = helper.make_graph(
        [node],
        "logcumsumexp",
        [helper.make_tensor_value_info("X", elem_type, shape)],
        [helper.make_tensor_value_info("Y", elem_type, shape)],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 29)])


@pytest.mark.parametrize("expanded", [False, True])
@pytest.mark.parametrize(
    "dtype", [np.float16, ml_dtypes.bfloat16, np.float32, np.float64]
)
@pytest.mark.parametrize("axis", [0, 1, 2, -1, -3])
def test_logcumsumexp_numerical(dtype, axis: int, expanded: bool) -> None:
    x = np.random.default_rng(42).uniform(-200, 200, (2, 3, 4)).astype(dtype)
    model = _model(x, axis)
    evaluator = ReferenceEvaluator(model, new_ops=[LogCumSumExp] if expanded else None)
    result = evaluator.run(None, {"X": x})[0]
    expected = _oracle(x, axis)
    assert result.dtype == x.dtype
    assert result.shape == x.shape
    assert_allclose(
        result.astype(np.float64), expected.astype(np.float64), rtol=2e-6, atol=2e-6
    )


@pytest.mark.parametrize("expanded", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "values",
    [
        [-200, -100, 100],
        [1000, 1001, -1000],
        [-np.inf, -np.inf, -np.inf],
        [-np.inf, 1, -np.inf, 2],
        [1, np.inf, 2, -np.inf, np.inf],
        [np.inf, -np.inf, np.nan, 1],
        [-np.inf, np.nan, np.inf],
        [np.nan, 1, 2],
        [0, -20, -40],
    ],
)
def test_logcumsumexp_prefixes(values, dtype, expanded: bool) -> None:
    x = np.array(values, dtype=dtype)
    evaluator = ReferenceEvaluator(
        _model(x, None), new_ops=[LogCumSumExp] if expanded else None
    )
    with np.errstate(invalid="raise"):
        result = evaluator.run(None, {"X": x})[0]
    tolerance = 2e-6 if dtype == np.float32 else 2e-14
    assert_allclose(result, _oracle(x, 0), rtol=tolerance, atol=0, equal_nan=True)


@pytest.mark.parametrize(
    "dtype", [np.float16, ml_dtypes.bfloat16, np.float32, np.float64]
)
def test_logcumsumexp_direct_and_expanded(dtype) -> None:
    x = np.array(
        [[-np.inf, -np.inf, 100, np.inf, np.nan], [-200, -100, 100, -np.inf, np.inf]],
        dtype=dtype,
    ).T
    model = _model(x, 0)
    direct = ReferenceEvaluator(model).run(None, {"X": x})[0]
    expanded = ReferenceEvaluator(model, new_ops=[LogCumSumExp]).run(None, {"X": x})[0]
    assert_allclose(
        direct.astype(np.float64),
        expanded.astype(np.float64),
        rtol=2e-6,
        equal_nan=True,
    )


@pytest.mark.parametrize("expanded", [False, True])
@pytest.mark.parametrize("shape,axis", [((0,), 0), ((2, 0, 3), 1), ((0, 3), -1)])
def test_logcumsumexp_empty(shape, axis: int, expanded: bool) -> None:
    x = np.empty(shape, dtype=np.float32)
    evaluator = ReferenceEvaluator(
        _model(x, axis), new_ops=[LogCumSumExp] if expanded else None
    )
    result = evaluator.run(None, {"X": x})[0]
    assert_array_equal(result, x)
    assert result.dtype == x.dtype


@pytest.mark.parametrize("axis", [None, 0, -1])
@pytest.mark.parametrize(
    "shape", [(2, 3), (0, 3, 4, 2), (2, 0, 4, 3), (2, 3, 0, 4), (2, 3, 4, 0)]
)
def test_logcumsumexp_inlining(shape, axis: int | None) -> None:
    x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    model = _model(x, axis)
    model.graph.input[0].CopyFrom(
        helper.make_tensor_value_info("X", TensorProto.FLOAT, shape)
    )
    expanded = inliner.inline_selected_functions(
        model, [("", "LogCumSumExp")], inline_schema_functions=True
    )
    assert all(node.op_type != "LogCumSumExp" for node in expanded.graph.node)
    checker.check_model(expanded)
    inferred = shape_inference.infer_shapes(expanded, strict_mode=True, data_prop=True)
    checker.check_model(inferred, full_check=True)
    assert (
        tuple(
            dim.dim_value for dim in inferred.graph.output[0].type.tensor_type.shape.dim
        )
        == shape
    )
    result = ReferenceEvaluator(inferred).run(None, {"X": x})[0]
    assert_allclose(result, _oracle(x, 0 if axis is None else axis), rtol=2e-6)


@pytest.mark.parametrize("dtype", [np.float16, ml_dtypes.bfloat16])
@pytest.mark.parametrize("expanded", [False, True])
def test_logcumsumexp_promoted_accumulation(dtype, expanded: bool) -> None:
    x = np.zeros(1024, dtype=dtype)
    evaluator = ReferenceEvaluator(
        _model(x, 0), new_ops=[LogCumSumExp] if expanded else None
    )
    result = evaluator.run(None, {"X": x})[0]
    expected = np.log(np.arange(1, x.size + 1, dtype=np.float64)).astype(dtype)
    # Float32 accumulation can move a result across a low-precision rounding
    # midpoint. Rounding the state at every prefix causes much larger drift.
    tolerance = 0.008 if dtype == ml_dtypes.bfloat16 else 0.001
    assert_allclose(
        result.astype(np.float64), expected.astype(np.float64), rtol=tolerance, atol=0
    )


@pytest.mark.parametrize("axis", [0, 1, -1, -3])
def test_logcumsumexp_unknown_rank(axis: int) -> None:
    x = np.arange(24, dtype=np.float32).reshape(2, 3, 4) - 12
    evaluator = ReferenceEvaluator(
        _model(x, axis, unknown_rank=True), new_ops=[LogCumSumExp]
    )
    for shape in ((2, 3, 4), (1, 5, 2, 3)):
        x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) - 12
        assert_allclose(evaluator.run(None, {"X": x})[0], _oracle(x, axis), rtol=2e-6)


@pytest.mark.parametrize("shape,axis", [((), 0), ((2, 3), 2), ((2, 3), -3)])
def test_logcumsumexp_invalid_axis(shape, axis: int) -> None:
    x = np.zeros(shape, dtype=np.float32)
    with pytest.raises(ValueError, match="LogCumSumExp"):
        ReferenceEvaluator(_model(x, axis)).run(None, {"X": x})


@pytest.mark.parametrize(
    "dtype",
    [TensorProto.FLOAT16, TensorProto.BFLOAT16, TensorProto.FLOAT, TensorProto.DOUBLE],
)
def test_logcumsumexp_expansion_checker(dtype: int) -> None:
    x_type = helper.make_tensor_type_proto(dtype, None)
    schema = defs.get_schema("LogCumSumExp", 29)
    node = helper.make_node("LogCumSumExp", ["X"], ["Y"], axis=-1)
    body = FunctionProto.FromString(
        schema.get_context_dependent_function(
            node.SerializeToString(), [x_type.SerializeToString()]
        )
    )
    checker.check_function(body)
