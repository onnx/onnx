# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
import pytest

from onnx import (
    TensorProto,
    checker,
    defs,
    helper,
    numpy_helper,
    shape_inference,
    version_converter,
)
from onnx.reference import ReferenceEvaluator
from onnx.reference.op_run import OpRun

_INT32_INDICES_OPSET = 29


def _model(op, data_shape, indices_shape, index_type, *, version=29, **attrs):
    inputs = [
        helper.make_tensor_value_info("data", TensorProto.FLOAT, data_shape),
        helper.make_tensor_value_info("indices", index_type, indices_shape),
    ]
    if op == "ScatterND":
        updates_shape = list(indices_shape[:-1]) + list(data_shape[indices_shape[-1] :])
        inputs.append(
            helper.make_tensor_value_info("updates", TensorProto.FLOAT, updates_shape)
        )
    graph = helper.make_graph(
        [helper.make_node(op, [value.name for value in inputs], ["output"], **attrs)],
        "nd_indices",
        inputs,
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, None)],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", version)])


@pytest.mark.parametrize("op", ["GatherND", "ScatterND"])
@pytest.mark.parametrize("version", [11, 12, 13, 16, 18, 28, 29])
@pytest.mark.parametrize(
    "index_type", [TensorProto.INT32, TensorProto.INT64, TensorProto.FLOAT]
)
def test_nd_indices_type_constraints(op, version, index_type):
    model = _model(op, [2, 3], [1, 1], index_type, version=version)
    if index_type == TensorProto.FLOAT or (
        version < _INT32_INDICES_OPSET and index_type == TensorProto.INT32
    ):
        with pytest.raises(shape_inference.InferenceError):
            shape_inference.infer_shapes(model, check_type=True, strict_mode=True)
    else:
        inferred = shape_inference.infer_shapes(
            model, check_type=True, strict_mode=True
        )
        checker.check_model(inferred, full_check=True)


@pytest.mark.parametrize("index_type", [TensorProto.INT32, TensorProto.INT64])
@pytest.mark.parametrize(
    ("data_shape", "indices_shape", "batch_dims", "output_shape"),
    [
        (["B", "S", "D"], ["B", "Q", "K", 1], 1, ["B", "Q", "K", "D"]),
        ([2, 3, 4], [2, 0, 1], 1, [2, 0, 4]),
        ([0, 3, 4], [0, 2, 1], 1, [0, 2, 4]),
        ([2, 3, 4], [2, 3, 2, 1], 2, [2, 3, 2]),
        ([2, 3], [2], 0, []),
        ([2, None, 4], [3, 1], 0, [3, None, 4]),
    ],
)
def test_gathernd_inference(
    index_type, data_shape, indices_shape, batch_dims, output_shape
):
    model = _model(
        "GatherND", data_shape, indices_shape, index_type, batch_dims=batch_dims
    )
    inferred = shape_inference.infer_shapes(model, check_type=True, strict_mode=True)
    expected = helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)
    actual = inferred.graph.output[0]
    for dim in actual.type.tensor_type.shape.dim:
        if dim.dim_param.startswith("unk__"):
            dim.ClearField("dim_param")
    assert actual == expected


@pytest.mark.parametrize("index_type", [TensorProto.INT32, TensorProto.INT64])
def test_scatternd_symbolic_inference(index_type):
    model = _model("ScatterND", ["B", "T", "D"], ["N", 2], index_type, reduction="add")
    inferred = shape_inference.infer_shapes(model, check_type=True, strict_mode=True)
    assert inferred.graph.output[0] == helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, ["B", "T", "D"]
    )


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
@pytest.mark.parametrize("batch_size", [0, 2])
@pytest.mark.parametrize("selection_size", [0, 2])
def test_gathernd_empty_and_batched_reference(dtype, batch_size, selection_size):
    data = np.arange(batch_size * 12, dtype=np.float32).reshape(batch_size, 4, 3)
    indices = np.empty((batch_size, 1, selection_size, 1), dtype=dtype)
    if selection_size:
        indices[...] = np.array([0, -1], dtype=dtype).reshape(1, 1, 2, 1)
    model = _model(
        "GatherND",
        data.shape,
        indices.shape,
        helper.np_dtype_to_tensor_dtype(np.dtype(dtype)),
        batch_dims=1,
    )
    output = ReferenceEvaluator(model).run(None, {"data": data, "indices": indices})[0]
    expected = data[:, np.array([0, 3])[:selection_size], :].reshape(
        batch_size, 1, selection_size, 3
    )
    np.testing.assert_array_equal(output, expected)


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_gathernd_scalar_and_bounds(dtype):
    data = np.arange(6, dtype=np.float32).reshape(2, 3)
    model = _model(
        "GatherND", data.shape, [2], helper.np_dtype_to_tensor_dtype(np.dtype(dtype))
    )
    evaluator = ReferenceEvaluator(model)
    output = evaluator.run(
        None, {"data": data, "indices": np.array([-1, -1], dtype=dtype)}
    )[0]
    np.testing.assert_array_equal(output, np.array(5, dtype=np.float32))
    for indices in ([2, 0], [-3, 0]):
        with pytest.raises(IndexError):
            evaluator.run(
                None, {"data": data, "indices": np.array(indices, dtype=dtype)}
            )


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_gathernd_multiple_batch_dims(dtype):
    data = np.arange(120, dtype=np.float32).reshape(2, 3, 4, 5)
    indices = np.broadcast_to(np.array([[0], [-1]], dtype=dtype), (2, 3, 2, 1))
    model = _model(
        "GatherND",
        data.shape,
        indices.shape,
        helper.np_dtype_to_tensor_dtype(np.dtype(dtype)),
        batch_dims=2,
    )
    output = ReferenceEvaluator(model).run(None, {"data": data, "indices": indices})[0]
    np.testing.assert_array_equal(output, data[:, :, [0, -1], :])


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
@pytest.mark.parametrize("reduction", ["none", "add", "mul", "max", "min"])
@pytest.mark.parametrize("empty", [False, True])
def test_scatternd_reductions_and_downgrade(dtype, reduction, empty):
    data = np.ones((2, 3, 2), dtype=np.float32)
    indices = np.array([[0, 1], [1, -1], [0, -2]], dtype=dtype)
    updates = np.array([[2, 3], [4, 5], [6, 7]], dtype=np.float32)
    if reduction == "none":
        indices = indices[:2]
        updates = updates[:2]
    if empty:
        indices, updates = indices[:0], updates[:0]
    model = _model(
        "ScatterND",
        data.shape,
        indices.shape,
        helper.np_dtype_to_tensor_dtype(np.dtype(dtype)),
        reduction=reduction,
    )
    expected = data.copy()
    index = tuple(indices.T)
    if reduction == "none":
        expected[index] = updates
    else:
        ufunc = {
            "add": np.add,
            "mul": np.multiply,
            "max": np.maximum,
            "min": np.minimum,
        }[reduction]
        ufunc.at(expected, index, updates)
    feeds = {"data": data, "indices": indices, "updates": updates}
    for version in (29, 28, 18):
        converted = version_converter.convert_version(model, version)
        converted = shape_inference.infer_shapes(converted, strict_mode=True)
        checker.check_model(converted, full_check=True)
        output = ReferenceEvaluator(converted).run(None, feeds)[0]
        np.testing.assert_array_equal(output, expected)


def test_scatternd_schema_versions():
    versions = [
        schema.since_version
        for schema in defs.get_all_schemas_with_history()
        if schema.name == "ScatterND" and schema.domain == ""
    ]
    assert versions.count(18) == versions.count(29) == 1
    previous = defs.get_schema("ScatterND", 18)
    latest = defs.get_schema("ScatterND", 29)
    assert previous.inputs[1].type_str == "tensor(int64)"
    assert "'mean'" not in previous.attributes["reduction"].description
    assert "'mean'" in latest.attributes["reduction"].description
    assert "include_self" not in latest.attributes
    assert latest.inputs[1].type_str == "Tind"
    assert set(
        next(
            constraint.allowed_type_strs
            for constraint in latest.type_constraints
            if constraint.type_param_str == "Tind"
        )
    ) == {"tensor(int32)", "tensor(int64)"}


@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
@pytest.mark.parametrize("value_dtype", [np.float64, np.int64, np.uint64])
@pytest.mark.parametrize("whole_tensor", [False, True])
def test_scatternd_mean_indices(index_dtype, value_dtype, whole_tensor):
    if value_dtype == np.uint64:
        values = [np.iinfo(np.uint64).max] * 3
        expected = values[0]
    else:
        values = [-2, -3, -3]
        expected = -8 / 3 if value_dtype == np.float64 else -3
    if whole_tensor:
        data = np.array([values[0]], dtype=value_dtype)
        indices = np.empty((2, 0), dtype=index_dtype)
        updates = np.array(values[1:], dtype=value_dtype).reshape(2, 1)
    else:
        data = np.array([[values[0]], [7]], dtype=value_dtype)
        indices = np.array([[0], [-2]], dtype=index_dtype)
        updates = np.array(values[1:], dtype=value_dtype).reshape(2, 1)
    model = _model(
        "ScatterND",
        data.shape,
        indices.shape,
        helper.np_dtype_to_tensor_dtype(np.dtype(index_dtype)),
        reduction="mean",
    )
    value_type = helper.np_dtype_to_tensor_dtype(np.dtype(value_dtype))
    for value in (model.graph.input[0], model.graph.input[2], model.graph.output[0]):
        value.type.tensor_type.elem_type = value_type
    inferred = shape_inference.infer_shapes(model, check_type=True, strict_mode=True)
    checker.check_model(inferred, full_check=True)
    result = ReferenceEvaluator(inferred).run(
        None, {"data": data, "indices": indices, "updates": updates}
    )[0]
    assert result.dtype == data.dtype
    if value_dtype == np.float64:
        np.testing.assert_allclose(result.flat[0], expected)
    else:
        assert result.flat[0] == expected
    if not whole_tensor:
        assert result[1, 0] == data[1, 0]
    with pytest.raises(RuntimeError, match="mean"):
        version_converter.convert_version(inferred, 28)
    assert [node.op_type for node in inferred.graph.node] == ["ScatterND"]


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
@pytest.mark.parametrize(
    "values, expected",
    [
        ([np.finfo(np.float64).max] * 3, np.finfo(np.float64).max),
        ([-np.finfo(np.float64).max] * 3, -np.finfo(np.float64).max),
        ([np.finfo(np.float64).max] * 2 + [-np.inf], -np.inf),
        ([-np.finfo(np.float64).max] * 2 + [np.inf], np.inf),
        ([np.inf, -np.inf, 1], np.nan),
        ([np.nan, 1, 2], np.nan),
    ],
)
def test_scatternd_mean_indices_nonfinite(dtype, values, expected):
    model = _model(
        "ScatterND",
        [2],
        [2, 1],
        helper.np_dtype_to_tensor_dtype(np.dtype(dtype)),
        reduction="mean",
    )
    for value in (model.graph.input[0], model.graph.input[2], model.graph.output[0]):
        value.type.tensor_type.elem_type = TensorProto.DOUBLE
    inferred = shape_inference.infer_shapes(model, check_type=True, strict_mode=True)
    checker.check_model(inferred, full_check=True)
    result = ReferenceEvaluator(inferred).run(
        None,
        {
            "data": np.array([values[0], -0.0]),
            "indices": np.array([[0], [-2]], dtype=dtype),
            "updates": np.array(values[1:], dtype=np.float64),
        },
    )[0]
    np.testing.assert_allclose(result, [expected, -0.0], equal_nan=True)
    assert np.signbit(result[1])


@pytest.mark.parametrize("initializer", [False, True])
@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_gathernd_downgrade_preserves_input_and_batch_dims(initializer, dtype):
    data = np.arange(24, dtype=np.float32).reshape(2, 4, 3)
    indices = np.array([[[0], [-1]], [[2], [1]]], dtype=dtype)
    model = _model(
        "GatherND",
        data.shape,
        indices.shape,
        helper.np_dtype_to_tensor_dtype(np.dtype(dtype)),
        batch_dims=1,
    )
    feeds = {"data": data, "indices": indices}
    if initializer:
        model.graph.initializer.append(numpy_helper.from_array(indices, "indices"))
        del model.graph.input[1]
        del feeds["indices"]
    converted = version_converter.convert_version(model, 13)
    converted = shape_inference.infer_shapes(converted, strict_mode=True)
    checker.check_model(converted, full_check=True)
    casts = [node for node in converted.graph.node if node.op_type == "Cast"]
    assert len(casts) == (1 if dtype == np.int32 else 0)
    if not initializer:
        assert converted.graph.input[1] == model.graph.input[1]
    expected = np.stack([data[0, [0, -1]], data[1, [2, 1]]])
    output = ReferenceEvaluator(converted).run(None, feeds)[0]
    np.testing.assert_array_equal(output, expected)


@pytest.mark.parametrize("op", ["GatherND", "ScatterND"])
@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_nd_indices_downgrade_unknown_intermediate_type(op, dtype):
    class OpaqueIndices(OpRun):
        op_domain = "test.nd_indices"

        def _run(self, indices):
            return (indices,)

    model = _model(op, [2, 3], [1, 1], helper.np_dtype_to_tensor_dtype(np.dtype(dtype)))
    model.graph.node[0].input[1] = "untyped_indices"
    model.graph.node.insert(
        0,
        helper.make_node(
            "OpaqueIndices",
            ["indices"],
            ["untyped_indices"],
            domain=OpaqueIndices.op_domain,
        ),
    )
    model.opset_import.append(helper.make_opsetid(OpaqueIndices.op_domain, 1))
    model.graph.output[0].CopyFrom(
        helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [1, 3] if op == "GatherND" else [2, 3]
        )
    )
    inferred = shape_inference.infer_shapes(model, strict_mode=True)
    assert not any(
        value.name == "untyped_indices" and value.type.tensor_type.elem_type
        for value in inferred.graph.value_info
    )
    converted = version_converter.convert_version(model, 28)
    assert [node.op_type for node in converted.graph.node] == [
        "OpaqueIndices",
        "Cast",
        op,
    ]
    cast_node = converted.graph.node[1]
    assert list(cast_node.input) == ["untyped_indices"]
    assert helper.get_attribute_value(cast_node.attribute[0]) == TensorProto.INT64
    assert converted.graph.node[2].input[1] == cast_node.output[0]
    inferred = shape_inference.infer_shapes(converted, strict_mode=True)
    checker.check_model(inferred, full_check=True)
    data = np.arange(6, dtype=np.float32).reshape(2, 3)
    feeds = {"data": data, "indices": np.array([[-1]], dtype=dtype)}
    if op == "GatherND":
        expected = data[[-1]]
    else:
        feeds["updates"] = np.array([[7, 8, 9]], dtype=np.float32)
        expected = np.array([[0, 1, 2], [7, 8, 9]], dtype=np.float32)
    actual = ReferenceEvaluator(converted, new_ops=[OpaqueIndices]).run(None, feeds)[0]
    np.testing.assert_array_equal(actual, expected)
