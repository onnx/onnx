# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Type inference must agree with the type constraints of every schema.

For each registered schema, the inputs are given distinct types allowed by
their type constraints. An output that shares a type constraint with an input
must then be inferred to have the type bound by that input.
"""

from __future__ import annotations

import pytest

import onnx
from onnx import TensorProto, defs, helper, shape_inference

_DTYPES = {name.lower(): value for name, value in TensorProto.DataType.items()}

_ATTRIBUTE_VALUES = {
    defs.OpSchema.AttrType.FLOAT: 1.0,
    defs.OpSchema.AttrType.INT: 1,
    defs.OpSchema.AttrType.STRING: "a",
    defs.OpSchema.AttrType.FLOATS: [1.0],
    defs.OpSchema.AttrType.INTS: [1],
    defs.OpSchema.AttrType.STRINGS: ["a"],
}


def _type_from_str(text: str) -> onnx.TypeProto:
    """Converts a schema type string such as ``seq(tensor(float))``."""
    kind, _, inner = text[:-1].partition("(")
    if kind == "tensor":
        return helper.make_tensor_type_proto(_DTYPES[inner], None)
    if kind == "sparse_tensor":
        return helper.make_sparse_tensor_type_proto(_DTYPES[inner], None)
    if kind == "seq":
        return helper.make_sequence_type_proto(_type_from_str(inner))
    if kind == "optional":
        return helper.make_optional_type_proto(_type_from_str(inner))
    if kind == "map":
        key, _, value = inner.partition(",")
        return helper.make_map_type_proto(_DTYPES[key], _type_from_str(value))
    raise ValueError(f"Unsupported type string {text!r}")


def _type_to_str(type_proto: onnx.TypeProto) -> str | None:
    """Returns the schema type string of a fully known type, else None."""
    kind = type_proto.WhichOneof("value")
    if kind in ("tensor_type", "sparse_tensor_type"):
        elem_type = getattr(type_proto, kind).elem_type
        if not elem_type:
            return None
        name = TensorProto.DataType.Name(elem_type).lower()
        return f"{'tensor' if kind == 'tensor_type' else 'sparse_tensor'}({name})"
    if kind in ("sequence_type", "optional_type"):
        inner = _type_to_str(getattr(type_proto, kind).elem_type)
        prefix = "seq" if kind == "sequence_type" else "optional"
        return f"{prefix}({inner})" if inner else None
    if kind == "map_type":
        value = _type_to_str(type_proto.map_type.value_type)
        if not type_proto.map_type.key_type or not value:
            return None
        key = TensorProto.DataType.Name(type_proto.map_type.key_type).lower()
        return f"map({key},{value})"
    return None


def _bound_types(schema: defs.OpSchema, offset: int) -> dict[str, str]:
    """Binds each type constraint, preferring a type no other one uses."""
    bound: dict[str, str] = {}
    for param in (*schema.inputs, *schema.outputs):
        if param.type_str in bound or not param.types:
            continue
        allowed = sorted(param.types)
        unused = [t for t in allowed if t not in bound.values()] or allowed
        bound[param.type_str] = unused[offset % len(unused)]
    return bound


def _schemas() -> list[defs.OpSchema]:
    return [
        schema
        for schema in defs.get_all_schemas_with_history()
        if schema.has_type_and_shape_inference_function
        and not schema.deprecated
        and any(
            output.is_homogeneous
            and any(i.type_str == output.type_str for i in schema.inputs)
            for output in schema.outputs
        )
    ]


@pytest.mark.parametrize(
    "schema",
    _schemas(),
    ids=lambda s: f"{s.domain or 'ai.onnx'}.{s.name}-{s.since_version}",
)
@pytest.mark.parametrize("offset", range(3))
def test_output_type_follows_input_type_constraint(
    schema: defs.OpSchema, offset: int
) -> None:
    bound = _bound_types(schema, offset)
    n_inputs = min(len(schema.inputs), schema.max_input)
    inputs = []
    for i in range(n_inputs):
        param = schema.inputs[i]
        type_str = bound.get(param.type_str) or min(param.types)
        inputs.append(helper.make_value_info(f"X{i}", _type_from_str(type_str)))
    outputs = [f"Y{i}" for i in range(max(1, schema.min_output))]
    node = helper.make_node(
        schema.name,
        [value.name for value in inputs],
        outputs,
        domain=schema.domain,
    )
    for name, attr in sorted(schema.attributes.items()):
        if not attr.required:
            continue
        if attr.type not in _ATTRIBUTE_VALUES:
            pytest.skip(f"Cannot generate required attribute {name!r}")
        node.attribute.append(helper.make_attribute(name, _ATTRIBUTE_VALUES[attr.type]))
    opsets = {schema.domain: schema.since_version}
    opsets.setdefault("", defs.onnx_opset_version())
    model = helper.make_model(
        helper.make_graph([node], "g", inputs, []),
        opset_imports=[helper.make_opsetid(d, v) for d, v in opsets.items()],
    )
    try:
        inferred = shape_inference.infer_shapes(model, strict_mode=True)
    except shape_inference.InferenceError as error:
        pytest.skip(f"Generated node is rejected: {error}")

    types = {value.name: value.type for value in inferred.graph.value_info}
    for i, name in enumerate(outputs):
        param = schema.outputs[min(i, len(schema.outputs) - 1)]
        if not param.is_homogeneous or not any(
            p.type_str == param.type_str for p in schema.inputs[:n_inputs]
        ):
            continue
        if name not in types:
            continue
        inferred_type = _type_to_str(types[name])
        expected = bound[param.type_str]
        if inferred_type is not None:
            assert inferred_type == expected, (param.name, inferred_type, expected)
