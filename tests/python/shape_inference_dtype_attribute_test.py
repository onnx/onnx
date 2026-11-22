# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from onnx import (
    AttributeProto,
    ModelProto,
    NodeProto,
    TensorProto,
    ValueInfoProto,
    defs,
    helper,
    shape_inference,
)

# Adding 2**32 keeps the low 32 bits, so narrowing such an attribute value to
# int32 would turn it back into the valid data type it was added to.
_WRAP = 2**32


def _versions(op: str, attribute: str, domain: str = "") -> list[int]:
    return sorted(
        schema.since_version
        for schema in defs.get_all_schemas_with_history()
        if schema.name == op
        and schema.domain == domain
        and attribute in schema.attributes
        and schema.attributes[attribute].type == AttributeProto.INT
    )


def _model(
    node: NodeProto,
    inputs: list[ValueInfoProto],
    version: int,
    initializers: list[TensorProto] | None = None,
    extra_opsets: list[tuple[str, int]] | None = None,
) -> ModelProto:
    graph = helper.make_graph(
        [node],
        "dtype_attribute",
        inputs,
        [helper.make_empty_tensor_value_info("y")],
        initializer=initializers or [],
    )
    opsets = [helper.make_opsetid("", version)]
    opsets += [helper.make_opsetid(domain, v) for domain, v in extra_opsets or []]
    return helper.make_model(graph, opset_imports=opsets)


def _check(model: ModelProto, wrap: int, expected_elem_type: int) -> None:
    if wrap:
        with pytest.raises(shape_inference.InferenceError):
            shape_inference.infer_shapes(model, strict_mode=True)
    else:
        inferred = shape_inference.infer_shapes(model, strict_mode=True)
        output_type = inferred.graph.output[0].type.tensor_type
        assert output_type.elem_type == expected_elem_type


def _float_input(name: str = "x", shape: tuple[int, ...] = (2, 3)) -> ValueInfoProto:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


@pytest.mark.parametrize("wrap", [0, _WRAP])
@pytest.mark.parametrize("version", _versions("Cast", "to"))
def test_cast_to(version: int, wrap: int) -> None:
    node = helper.make_node("Cast", ["x"], ["y"], to=TensorProto.FLOAT16 + wrap)
    _check(_model(node, [_float_input()], version), wrap, TensorProto.FLOAT16)


@pytest.mark.parametrize("wrap", [0, _WRAP])
@pytest.mark.parametrize("version", _versions("BitCast", "to"))
def test_bitcast_to(version: int, wrap: int) -> None:
    node = helper.make_node("BitCast", ["x"], ["y"], to=TensorProto.INT32 + wrap)
    _check(_model(node, [_float_input()], version), wrap, TensorProto.INT32)


@pytest.mark.parametrize("wrap", [0, _WRAP])
@pytest.mark.parametrize("version", _versions("Multinomial", "dtype"))
def test_multinomial_dtype(version: int, wrap: int) -> None:
    node = helper.make_node("Multinomial", ["x"], ["y"], dtype=TensorProto.INT32 + wrap)
    _check(_model(node, [_float_input()], version), wrap, TensorProto.INT32)


@pytest.mark.parametrize("wrap", [0, _WRAP])
@pytest.mark.parametrize("version", _versions("Bernoulli", "dtype"))
def test_bernoulli_dtype(version: int, wrap: int) -> None:
    node = helper.make_node("Bernoulli", ["x"], ["y"], dtype=TensorProto.DOUBLE + wrap)
    _check(_model(node, [_float_input()], version), wrap, TensorProto.DOUBLE)


@pytest.mark.parametrize("wrap", [0, _WRAP])
@pytest.mark.parametrize("version", _versions("QuantizeLinear", "output_dtype"))
def test_quantize_linear_output_dtype_matching_zero_point(
    version: int, wrap: int
) -> None:
    # With a zero point, the output type comes from the zero point once
    # output_dtype matches it, so a wrapped value must not count as a match.
    node = helper.make_node(
        "QuantizeLinear",
        ["x", "scale", "zero_point"],
        ["y"],
        output_dtype=TensorProto.UINT8 + wrap,
    )
    initializers = [
        helper.make_tensor("scale", TensorProto.FLOAT, [], [1.0]),
        helper.make_tensor("zero_point", TensorProto.UINT8, [], [0]),
    ]
    model = _model(node, [_float_input()], version, initializers)
    _check(model, wrap, TensorProto.UINT8)


@pytest.mark.parametrize("wrap", [0, _WRAP])
@pytest.mark.parametrize("version", _versions("DequantizeLinear", "output_dtype"))
def test_dequantize_linear_output_dtype(version: int, wrap: int) -> None:
    node = helper.make_node(
        "DequantizeLinear",
        ["x", "scale"],
        ["y"],
        output_dtype=TensorProto.FLOAT + wrap,
    )
    inputs = [helper.make_tensor_value_info("x", TensorProto.UINT8, (2, 3))]
    initializers = [helper.make_tensor("scale", TensorProto.FLOAT, [], [1.0])]
    _check(_model(node, inputs, version, initializers), wrap, TensorProto.FLOAT)


def test_dequantize_linear_output_dtype_wrapping_to_undefined() -> None:
    # 2**32 would narrow to UNDEFINED, which means "take the type of the scale".
    node = helper.make_node(
        "DequantizeLinear", ["x", "scale"], ["y"], output_dtype=_WRAP
    )
    inputs = [helper.make_tensor_value_info("x", TensorProto.UINT8, (2, 3))]
    initializers = [helper.make_tensor("scale", TensorProto.FLOAT, [], [1.0])]
    version = _versions("DequantizeLinear", "output_dtype")[-1]
    _check(_model(node, inputs, version, initializers), _WRAP, TensorProto.FLOAT)


@pytest.mark.parametrize("wrap", [0, _WRAP])
def test_flex_attention_softmax_precision(wrap: int) -> None:
    domain = "ai.onnx.preview"
    node = helper.make_node(
        "FlexAttention",
        ["q", "k", "v"],
        ["y"],
        domain=domain,
        softmax_precision=TensorProto.FLOAT + wrap,
    )
    inputs = [
        _float_input("q", (1, 1, 2, 4)),
        _float_input("k", (1, 1, 3, 4)),
        _float_input("v", (1, 1, 3, 4)),
    ]
    model = _model(node, inputs, defs.onnx_opset_version(), extra_opsets=[(domain, 1)])
    _check(model, wrap, TensorProto.FLOAT)
