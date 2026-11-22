# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from onnx import ModelProto, TensorProto, defs, helper, shape_inference

_AXIS_OPSETS = [
    ("Concat", [4, 11, 13]),
    ("Gather", [1, 11, 13]),
    ("Compress", [11, 28]),
    ("Unique", [11, 28]),
    ("ConcatFromSequence", [11]),
    ("Softmax", [11, 13]),
    ("LogSoftmax", [11, 13]),
    ("Hardmax", [11, 13]),
    ("SplitToSequence", [11, 24]),
    ("OneHot", [9, 11, 28]),
    ("DFT", [17]),
]


def _negative_axis_skips_shape(op: str, version: int) -> bool:
    # Concat before opset 11 does not define negative axes; inference leaves the
    # output shape unknown instead of rejecting the model.
    return op == "Concat" and version < 11


def _axis_model(op: str, version: int, axis: int) -> ModelProto:
    inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])]
    node_inputs = ["x"]
    initializers = []
    if op == "Concat":
        node_inputs.append("x")
    elif op == "Gather":
        node_inputs.append("indices")
        initializers.append(helper.make_tensor("indices", TensorProto.INT64, [1], [0]))
    elif op == "Compress":
        node_inputs.append("condition")
        initializers.append(
            helper.make_tensor("condition", TensorProto.BOOL, [1], [True])
        )
    elif op == "ConcatFromSequence":
        inputs = [
            helper.make_tensor_sequence_value_info("x", TensorProto.FLOAT, [2, 3])
        ]
    elif op == "OneHot":
        inputs = [helper.make_tensor_value_info("x", TensorProto.INT64, [2, 3])]
        node_inputs.extend(["depth", "values"])
        initializers.extend(
            [
                helper.make_tensor("depth", TensorProto.INT64, [], [4]),
                helper.make_tensor("values", TensorProto.FLOAT, [2], [0, 1]),
            ]
        )
    elif op == "DFT":
        inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 1])]
    output = (
        helper.make_tensor_sequence_value_info("y", TensorProto.FLOAT, None)
        if op == "SplitToSequence"
        else helper.make_empty_tensor_value_info("y")
    )
    return helper.make_model(
        helper.make_graph(
            [helper.make_node(op, node_inputs, ["y"], axis=axis)],
            "int64_axis",
            inputs,
            [output],
            initializer=initializers,
        ),
        opset_imports=[helper.make_opsetid("", version)],
    )


@pytest.mark.parametrize(
    ("op", "version"),
    [(op, version) for op, versions in _AXIS_OPSETS for version in versions],
)
@pytest.mark.parametrize(
    "axis", [2**31 - 1, -(2**31), 2**32 + 1, -(2**32) + 1, 2**63 - 1, -(2**63)]
)
def test_operator_axis_does_not_wrap(op: str, version: int, axis: int) -> None:
    model = _axis_model(op, version, axis)
    if axis < 0 and _negative_axis_skips_shape(op, version):
        inferred = shape_inference.infer_shapes(model, strict_mode=True)
        assert not inferred.graph.output[0].type.tensor_type.HasField("shape")
        return
    with pytest.raises(shape_inference.InferenceError):
        shape_inference.infer_shapes(model, strict_mode=True)


@pytest.mark.parametrize(
    ("op", "version"),
    [(op, version) for op, versions in _AXIS_OPSETS for version in versions],
)
def test_operator_valid_axis_preserved(op: str, version: int) -> None:
    inferred = shape_inference.infer_shapes(
        _axis_model(op, version, 1), strict_mode=True
    )
    output_type = inferred.graph.output[0].type
    shape = (
        output_type.sequence_type.elem_type.tensor_type.shape
        if op == "SplitToSequence"
        else output_type.tensor_type.shape
    )
    expected = {
        "Concat": [2, 6],
        "Gather": [2, 1],
        "Compress": [2, None],
        "Unique": [2, None],
        "ConcatFromSequence": [2, None],
        "Softmax": [2, 3],
        "LogSoftmax": [2, 3],
        "Hardmax": [2, 3],
        "SplitToSequence": [2, 1],
        "OneHot": [2, 4, 3],
        "DFT": [2, 3, 2],
    }
    assert [
        dim.dim_value if dim.HasField("dim_value") else None for dim in shape.dim
    ] == expected[op]
    negative_axis = -2 if op in {"OneHot", "DFT"} else -1
    negative = shape_inference.infer_shapes(
        _axis_model(op, version, negative_axis), strict_mode=True
    )
    if _negative_axis_skips_shape(op, version):
        assert not negative.graph.output[0].type.tensor_type.HasField("shape")
    else:
        assert negative.graph.output[0].type == output_type


@pytest.mark.parametrize("new_axis", [2**32, 2**32 + 1, -(2**32), -(2**32) + 1])
def test_concat_from_sequence_new_axis_does_not_wrap(new_axis: int) -> None:
    model = _axis_model("ConcatFromSequence", 11, 1)
    model.graph.node[0].attribute.append(helper.make_attribute("new_axis", new_axis))
    with pytest.raises(shape_inference.InferenceError):
        shape_inference.infer_shapes(model, strict_mode=True)


@pytest.mark.parametrize("axis", [1, -2])
def test_concat_from_sequence_valid_new_axis(axis: int) -> None:
    model = _axis_model("ConcatFromSequence", 11, axis)
    model.graph.node[0].attribute.append(helper.make_attribute("new_axis", 1))
    inferred = shape_inference.infer_shapes(model, strict_mode=True)
    shape = inferred.graph.output[0].type.tensor_type.shape
    assert len(shape.dim) == 3
    assert shape.dim[0].dim_value == 2
    assert shape.dim[2].dim_value == 3
    assert not shape.dim[1].HasField("dim_value")


def _scan_model(version: int, attribute: str, axis: int) -> ModelProto:
    body = helper.make_graph(
        [helper.make_node("Identity", ["body_in"], ["body_out"])],
        "body",
        [helper.make_tensor_value_info("body_in", TensorProto.FLOAT, [None])],
        [helper.make_tensor_value_info("body_out", TensorProto.FLOAT, [None])],
    )
    node = helper.make_node("Scan", ["x"], ["y"], body=body, num_scan_inputs=1)
    node.attribute.append(helper.make_attribute(attribute, [axis]))
    return helper.make_model(
        helper.make_graph(
            [node],
            "scan_int64_axes",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])],
            [helper.make_empty_tensor_value_info("y")],
        ),
        opset_imports=[helper.make_opsetid("", version)],
    )


@pytest.mark.parametrize(
    "version",
    sorted(
        schema.since_version
        for schema in defs.get_all_schemas_with_history()
        if schema.name == "Scan" and schema.domain == "" and schema.since_version >= 9
    ),
)
@pytest.mark.parametrize("attribute", ["scan_input_axes", "scan_output_axes"])
@pytest.mark.parametrize("axis", [2**32 + 1, -(2**32) + 1, 2**63 - 1, -(2**63), 1, -1])
def test_scan_int64_axes(version: int, attribute: str, axis: int) -> None:
    model = _scan_model(version, attribute, axis)
    if axis in {1, -1}:
        inferred = shape_inference.infer_shapes(model, strict_mode=True)
        assert [
            dim.dim_value for dim in inferred.graph.output[0].type.tensor_type.shape.dim
        ] == [3, 2]
    else:
        with pytest.raises(shape_inference.InferenceError, match="axis"):
            shape_inference.infer_shapes(model, strict_mode=True)


def _split_model(
    version: int,
    dimension: int,
    *,
    axis: int = 1,
    split: list[int] | None = None,
    num_outputs: int = 2,
) -> ModelProto:
    outputs = [f"y{i}" for i in range(num_outputs)]
    node = helper.make_node("Split", ["x"], outputs, axis=axis)
    initializers = []
    if split is not None:
        if version < 13:
            node.attribute.append(helper.make_attribute("split", split))
        else:
            node.input.append("split")
            initializers.append(
                helper.make_tensor("split", TensorProto.INT64, [len(split)], split)
            )
    elif version >= 18:
        node.attribute.append(helper.make_attribute("num_outputs", num_outputs))
    graph = helper.make_graph(
        [node],
        "split_int64",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, dimension])],
        [helper.make_empty_tensor_value_info(name) for name in outputs],
        initializer=initializers,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", version)])


@pytest.mark.parametrize("version", [2, 11, 13, 18])
@pytest.mark.parametrize("dimension", [2**31, 2**32 + 10, 2**63 - 2])
@pytest.mark.parametrize("explicit_split", [False, True])
def test_split_large_dimension(
    version: int, dimension: int, explicit_split: bool
) -> None:
    lengths = [dimension // 2, dimension // 2]
    model = _split_model(version, dimension, split=lengths if explicit_split else None)
    inferred = shape_inference.infer_shapes(model, strict_mode=True)
    assert [
        [dim.dim_value for dim in output.type.tensor_type.shape.dim]
        for output in inferred.graph.output
    ] == [[1, length] for length in lengths]


@pytest.mark.parametrize(
    ("dimension", "lengths"),
    [
        (2**32 + 11, [2147483654, 2147483653]),
        (2**63 - 1, [2**62, 2**62 - 1]),
        (2**33 + 6, [2863311533, 2863311533, 2863311532]),
    ],
)
def test_split_large_uneven_dimension(dimension: int, lengths: list[int]) -> None:
    model = _split_model(18, dimension, num_outputs=len(lengths))
    inferred = shape_inference.infer_shapes(model, strict_mode=True)
    assert [
        output.type.tensor_type.shape.dim[1].dim_value
        for output in inferred.graph.output
    ] == lengths


@pytest.mark.parametrize(
    ("dimension", "lengths"), [(7, [3, 3, 1]), (4, [2, 2, 0]), (6, [2, 2, 2])]
)
def test_split_num_outputs_chunks(dimension: int, lengths: list[int]) -> None:
    model = _split_model(18, dimension, num_outputs=len(lengths))
    inferred = shape_inference.infer_shapes(model, strict_mode=True)
    assert [
        output.type.tensor_type.shape.dim[1].dim_value
        for output in inferred.graph.output
    ] == lengths


# ceil(3 / 5) = 1 leaves -1 for the last of 5 chunks; ceil(5 / 4) = 2 leaves -1.
@pytest.mark.parametrize(("dimension", "num_outputs"), [(3, 5), (5, 4)])
def test_split_num_outputs_negative_last_chunk(
    dimension: int, num_outputs: int
) -> None:
    model = _split_model(18, dimension, num_outputs=num_outputs)
    with pytest.raises(shape_inference.InferenceError, match="negative size"):
        shape_inference.infer_shapes(model, strict_mode=True)


def _split_model_with_num_outputs(
    dimension: int, num_outputs: int, node_outputs: int
) -> ModelProto:
    model = _split_model(18, dimension, num_outputs=node_outputs)
    attribute = next(
        a for a in model.graph.node[0].attribute if a.name == "num_outputs"
    )
    attribute.i = num_outputs
    return model


@pytest.mark.parametrize(
    ("dimension", "num_outputs", "lengths"),
    [(2**40, 2**40, [1, 1]), (0, 2**40, [0, 0]), (10, 3, [4, 4])],
)
def test_split_num_outputs_larger_than_node_outputs(
    dimension: int, num_outputs: int, lengths: list[int]
) -> None:
    # Only the node's outputs get a size, so a large 'num_outputs' must not
    # allocate one entry per chunk.
    model = _split_model_with_num_outputs(dimension, num_outputs, len(lengths))
    inferred = shape_inference.infer_shapes(model, strict_mode=True)
    assert [
        output.type.tensor_type.shape.dim[1].dim_value
        for output in inferred.graph.output
    ] == lengths


def test_split_num_outputs_larger_than_dimension() -> None:
    model = _split_model_with_num_outputs(10, 2**40, 2)
    with pytest.raises(shape_inference.InferenceError, match="negative size"):
        shape_inference.infer_shapes(model, strict_mode=True)


@pytest.mark.parametrize("version", [2, 11, 13, 18])
def test_split_lengths_sum_overflow(version: int) -> None:
    # This sum wraps to 10 with unchecked signed 64-bit arithmetic.
    model = _split_model(version, 10, split=[2**63 - 1, 2**63 - 1, 12], num_outputs=3)
    with pytest.raises(shape_inference.InferenceError, match="overflow"):
        shape_inference.infer_shapes(model, strict_mode=True)


@pytest.mark.parametrize("version", [2, 11, 13, 18])
@pytest.mark.parametrize("axis", [2**32 + 1, -(2**32) + 1, 2**63 - 1, -(2**63)])
def test_split_axis_does_not_wrap(version: int, axis: int) -> None:
    model = _split_model(version, 10, axis=axis)
    with pytest.raises(shape_inference.InferenceError, match="axis"):
        shape_inference.infer_shapes(model, strict_mode=True)


@pytest.mark.parametrize(
    "version",
    sorted(
        schema.since_version
        for schema in defs.get_all_schemas_with_history()
        if schema.name == "Flatten" and schema.domain == ""
    ),
)
@pytest.mark.parametrize("axis", [2**32 + 1, -(2**32) + 1, 2**63 - 1, -(2**63)])
def test_flatten_axis_does_not_wrap(version: int, axis: int) -> None:
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("Flatten", ["x"], ["y"], axis=axis)],
            "flatten_int64_axis",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4])],
            [helper.make_empty_tensor_value_info("y")],
        ),
        opset_imports=[helper.make_opsetid("", version)],
    )
    with pytest.raises(shape_inference.InferenceError, match="axis"):
        shape_inference.infer_shapes(model, strict_mode=True)


@pytest.mark.parametrize("version", [1, 11, 13, 18, 28])
@pytest.mark.parametrize("channels", [0, 4, 32768**2])
def test_depth_to_space_large_blocksize(version: int, channels: int) -> None:
    # 32768**2 fits in int32. Invalid channel counts must be rejected
    # instead of being rounded down to zero by integer division.
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("DepthToSpace", ["x"], ["y"], blocksize=32768)],
            "depth_to_space_large_blocksize",
            [
                helper.make_tensor_value_info(
                    "x", TensorProto.FLOAT, [1, channels, 2, 2]
                )
            ],
            [helper.make_empty_tensor_value_info("y")],
        ),
        opset_imports=[helper.make_opsetid("", version)],
    )
    if channels == 4:
        with pytest.raises(shape_inference.InferenceError, match="must be a multiple"):
            shape_inference.infer_shapes(model, strict_mode=True)
    else:
        inferred = shape_inference.infer_shapes(model, strict_mode=True)
        assert [
            dim.dim_value for dim in inferred.graph.output[0].type.tensor_type.shape.dim
        ] == [1, channels // 32768**2, 65536, 65536]
