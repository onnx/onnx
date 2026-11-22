# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from onnx import ModelProto, TensorProto, defs, helper, shape_inference


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
    # 32768**2 fits in int32. The report's zero-channel result is caused
    # by truncating division of an invalid channel count, not overflow.
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
