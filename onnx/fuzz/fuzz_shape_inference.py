# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Atheris fuzz harness for onnx.shape_inference.

Valid serialized models always take the raw path. If parsing fails, a
fuzzer-controlled toggle byte at the tail can select the structured fallback:

* Raw bytes  -> onnx.load_model_from_string  -> infer_shapes
  Catches protobuf parser bugs and bugs reachable only through crafted
  serialized models the structured builder will not produce.

* Structured -> helper.make_model from FuzzedDataProvider -> infer_shapes
  Chains nodes of registered operator schemas whose inputs are untyped,
  typed (arbitrarily or within the schema's type constraints), constant,
  omitted, or produced by an earlier node, with generated attributes; or
  builds graphs with subgraph-bearing ops (If / Loop / Scan) to reach the
  recursive visitor without relying on random bytes parsing as a model.

All combinations of strict_mode, check_type, and data_prop are sampled.

Besides crashes, a successful inference is a finding when inferring the
result again fails or changes an inferred type, or when a node whose input
types satisfy its schema is assigned an output type that differs from the
type its type constraint takes from those inputs.
"""

from __future__ import annotations

import math
import sys
from typing import TYPE_CHECKING

import atheris

if TYPE_CHECKING:
    from typing import Any

with atheris.instrument_imports():
    import onnx
    from onnx import TensorProto, defs, helper, shape_inference

# Elementwise unary ops with trivial shape inference rules. Useful as
# filler nodes so generated graphs have non-trivial bodies that exercise
# the per-op inference dispatch table.
_UNARY = (
    "Relu",
    "Sigmoid",
    "Tanh",
    "Abs",
    "Neg",
    "Exp",
    "Log",
    "Sqrt",
    "Identity",
    "Floor",
    "Ceil",
)

# Ops that carry one or more subgraph attributes. Each forces the recursive
# shape-inference visitor to descend, which is the path the known DoS lives
# on. Loop/Scan exercise different subgraph-context plumbing than If.
_SUBGRAPH_OPS = ("If", "Loop", "Scan")

# Include historical versions and ML operators, whose inference functions
# must also tolerate inputs with no known type.
_SCHEMAS = tuple(
    schema
    for schema in defs.get_all_schemas_with_history()
    if schema.domain in ("", "ai.onnx.ml")
    and schema.has_type_and_shape_inference_function
)


_ELEM_TYPES = tuple(
    value for value in TensorProto.DataType.values() if value != TensorProto.UNDEFINED
)
# Type strings use the lower-case data type names, e.g. "tensor(float16)".
_DTYPE_NAMES = {value: name.lower() for name, value in TensorProto.DataType.items()}
_DTYPE_VALUES = {name: value for value, name in _DTYPE_NAMES.items()}

# Kinds of producer for each schema input.
_UNTYPED, _TYPED, _INITIALIZER, _OMITTED, _PRODUCED = range(5)
_MAX_EXTRA_INPUTS = 2
_MAX_NODES = 3
_MAX_RANK = 4

_INT_EDGES = (-(2**63), 2**63 - 1, -(2**31) - 1, 2**31, 2**32)
_FLOAT_EDGES = (math.nan, math.inf, -math.inf, -0.0, 1e-45, 3.4e38)
_STRINGS: tuple[str | bytes, ...] = ("", "a", "\u00e9", b"\xff\xfe")


def _make_shape(fdp):
    rank = fdp.ConsumeIntInRange(-1, _MAX_RANK)
    if rank < 0:
        # No shape information at all.
        return None
    shape: list[int | str | None] = []
    for i in range(rank):
        kind = fdp.ConsumeIntInRange(0, 2)
        if kind == 0:
            shape.append(fdp.ConsumeIntInRange(0, 8))
        elif kind == 1:
            shape.append(f"d{i}")
        else:
            shape.append(None)
    return shape


def _make_tensor_type(fdp):
    elem_type = _ELEM_TYPES[fdp.ConsumeIntInRange(0, len(_ELEM_TYPES) - 1)]
    return helper.make_tensor_type_proto(elem_type, _make_shape(fdp))


def _type_from_str(text):
    """Converts a schema type string such as ``seq(tensor(float))``."""
    kind, _, inner = text[:-1].partition("(")
    if kind == "tensor":
        return helper.make_tensor_type_proto(_DTYPE_VALUES[inner], None)
    if kind == "sparse_tensor":
        return helper.make_sparse_tensor_type_proto(_DTYPE_VALUES[inner], None)
    if kind == "seq":
        return helper.make_sequence_type_proto(_type_from_str(inner))
    if kind == "optional":
        return helper.make_optional_type_proto(_type_from_str(inner))
    if kind == "map":
        key, _, value = inner.partition(",")
        return helper.make_map_type_proto(_DTYPE_VALUES[key], _type_from_str(value))
    raise ValueError(f"Unsupported type string {text!r}")


def _type_to_str(type_proto):
    """Returns the schema type string of a fully known type, else None."""
    kind = type_proto.WhichOneof("value") if type_proto is not None else None
    if kind in ("tensor_type", "sparse_tensor_type"):
        elem_type = getattr(type_proto, kind).elem_type
        name = _DTYPE_NAMES.get(elem_type) if elem_type else None
        prefix = "tensor" if kind == "tensor_type" else "sparse_tensor"
        return f"{prefix}({name})" if name else None
    if kind in ("sequence_type", "optional_type"):
        inner = _type_to_str(getattr(type_proto, kind).elem_type)
        prefix = "seq" if kind == "sequence_type" else "optional"
        return f"{prefix}({inner})" if inner else None
    if kind == "map_type":
        key = _DTYPE_NAMES.get(type_proto.map_type.key_type)
        value = _type_to_str(type_proto.map_type.value_type)
        return f"map({key},{value})" if key and value else None
    return None


def _make_schema_type(fdp, param):
    """Samples a type allowed by the formal parameter's type constraint."""
    allowed = sorted(param.types)
    type_proto = _type_from_str(allowed[fdp.ConsumeIntInRange(0, len(allowed) - 1)])
    if type_proto.HasField("tensor_type"):
        shape = _make_shape(fdp)
        if shape is not None:
            type_proto.tensor_type.shape.SetInParent()
            for dim in shape:
                proto = type_proto.tensor_type.shape.dim.add()
                if isinstance(dim, int):
                    proto.dim_value = dim
                elif isinstance(dim, str):
                    proto.dim_param = dim
    return type_proto


def _make_type(fdp):
    kind = fdp.ConsumeIntInRange(0, 7)
    if kind == 6:
        return helper.make_sequence_type_proto(_make_tensor_type(fdp))
    if kind == 7:
        key_type = (TensorProto.STRING, TensorProto.INT64)[fdp.ConsumeIntInRange(0, 1)]
        return helper.make_map_type_proto(key_type, _make_tensor_type(fdp))
    return _make_tensor_type(fdp)


def _make_tensor(fdp, name):
    """Builds a small constant tensor of any data type."""
    elem_type = _ELEM_TYPES[fdp.ConsumeIntInRange(0, len(_ELEM_TYPES) - 1)]
    dims = [fdp.ConsumeIntInRange(0, 3) for _ in range(fdp.ConsumeIntInRange(0, 2))]
    count = math.prod(dims)
    value = fdp.ConsumeIntInRange(0, 1)
    if elem_type == TensorProto.STRING:
        values: list[Any] = [_STRINGS[value]] * count
    elif elem_type in (TensorProto.COMPLEX64, TensorProto.COMPLEX128):
        values = [complex(value)] * count
    else:
        values = [value] * count
    return helper.make_tensor(name, elem_type, dims, values)


def _make_initializer(fdp, name):
    # Small constant INT64 vectors drive data-dependent inference (Reshape,
    # Expand, Slice, ...); other constants may have any data type and shape.
    if fdp.ConsumeBool():
        values = [
            fdp.ConsumeIntInRange(-2, 6) for _ in range(fdp.ConsumeIntInRange(0, 4))
        ]
        return helper.make_tensor(name, TensorProto.INT64, [len(values)], values)
    return _make_tensor(fdp, name)


def _small_ints(fdp):
    return [fdp.ConsumeIntInRange(-2, 4) for _ in range(fdp.ConsumeIntInRange(0, 3))]


def _make_int(fdp):
    kind = fdp.ConsumeIntInRange(0, 2)
    if kind == 0:
        return fdp.ConsumeIntInRange(-2, 4)
    if kind == 1:
        # Attributes such as "to" and "dtype" hold data type enum values.
        return fdp.ConsumeIntInRange(0, max(_ELEM_TYPES) + 1)
    return _INT_EDGES[fdp.ConsumeIntInRange(0, len(_INT_EDGES) - 1)]


def _make_float(fdp):
    if fdp.ConsumeBool():
        return _FLOAT_EDGES[fdp.ConsumeIntInRange(0, len(_FLOAT_EDGES) - 1)]
    return float(fdp.ConsumeIntInRange(-2, 4))


def _make_string(fdp, default):
    # Prefer the schema default, which is usually a valid enumeration value.
    if default is not None and fdp.ConsumeBool():
        return default
    return _STRINGS[fdp.ConsumeIntInRange(0, len(_STRINGS) - 1)]


def _make_attribute(fdp, name, attr_type, default=None):
    t = defs.OpSchema.AttrType
    if attr_type == t.INT:
        value: Any = _make_int(fdp)
    elif attr_type == t.INTS:
        value = _small_ints(fdp)
    elif attr_type == t.FLOAT:
        value = _make_float(fdp)
    elif attr_type == t.FLOATS:
        value = [float(v) for v in _small_ints(fdp)]
    elif attr_type == t.STRING:
        value = _make_string(fdp, default)
    elif attr_type == t.STRINGS:
        value = [_make_string(fdp, default) for _ in range(fdp.ConsumeIntInRange(0, 3))]
    elif attr_type == t.TENSOR:
        value = _make_tensor(fdp, name)
    elif attr_type == t.GRAPH:
        value = helper.make_graph([], name, [], [])
    else:
        return None
    return helper.make_attribute(name, value, attr_type=int(attr_type))


# Consistent attributes needed to reach the inference code in #8651 rather
# than stopping at an unrelated attribute error.
_FIXED_ATTRIBUTES: dict[tuple[str, str], dict[str, Any]] = {
    ("", "RegexFullMatch"): {"pattern": "a.*"},
    ("ai.onnx.ml", "DictVectorizer"): {"string_vocabulary": ["a"]},
    ("ai.onnx.ml", "LabelEncoder"): {"keys_strings": ["a"], "values_int64s": [1]},
    ("ai.onnx.ml", "TreeEnsemble"): {
        "nodes_splits": helper.make_tensor("s", TensorProto.FLOAT, [1], [0.5]),
        "nodes_featureids": [0],
        "nodes_modes": helper.make_tensor("m", TensorProto.UINT8, [1], [0]),
        "nodes_truenodeids": [0],
        "nodes_falsenodeids": [0],
        "nodes_trueleafs": [1],
        "nodes_falseleafs": [1],
        "leaf_targetids": [0],
        "leaf_weights": helper.make_tensor("w", TensorProto.FLOAT, [1], [1.0]),
        "tree_roots": [0],
        "n_targets": 1,
    },
}


def _fixed_attributes(schema) -> dict[str, Any]:
    if (schema.name, schema.since_version) == ("LabelEncoder", 1):
        return {"classes_strings": ["a"]}
    return _FIXED_ATTRIBUTES.get((schema.domain, schema.name), {})


def _schema_attributes(fdp, schema):
    # Supply required attributes so inference is not rejected up front, and
    # sample optional ones to reach attribute-dependent branches.
    attrs = {}
    for name, attr in sorted(schema.attributes.items()):
        if attr.required or fdp.ConsumeBool():
            default = attr.default_value
            proto = _make_attribute(
                fdp,
                name,
                attr.type,
                default.s if default.type == default.STRING else None,
            )
            if proto is not None:
                attrs[name] = proto
    for name, value in _fixed_attributes(schema).items():
        attrs[name] = helper.make_attribute(name, value)
    return list(attrs.values())


def _schema_input(fdp, schema, index, name, produced, graph_inputs, initializers):
    """Returns the node input name after creating its producer."""
    param = schema.inputs[min(index, len(schema.inputs) - 1)]
    kind = fdp.ConsumeIntInRange(_UNTYPED, _PRODUCED if produced else _OMITTED)
    if kind == _UNTYPED:
        graph_inputs.append(onnx.ValueInfoProto(name=name))
    elif kind == _TYPED:
        if param.types and fdp.ConsumeBool():
            type_proto = _make_schema_type(fdp, param)
        else:
            type_proto = _make_type(fdp)
        graph_inputs.append(helper.make_value_info(name, type_proto))
    elif kind == _INITIALIZER:
        initializers.append(_make_initializer(fdp, name))
    elif kind == _OMITTED:
        # An empty name marks an omitted optional input.
        return ""
    else:
        return produced[fdp.ConsumeIntInRange(0, len(produced) - 1)]
    return name


def _build_schema_model(fdp):
    nodes = []
    imports: dict[str, int] = {}
    graph_inputs = []
    initializers = []
    produced: list[str] = []
    for k in range(_MAX_NODES):
        schema = _SCHEMAS[fdp.ConsumeIntInRange(0, len(_SCHEMAS) - 1)]
        n_inputs = fdp.ConsumeIntInRange(
            schema.min_input,
            min(schema.max_input, schema.min_input + _MAX_EXTRA_INPUTS),
        )
        prefix = "" if k == 0 else f"N{k}_"
        inputs = [f"{prefix}X{i}" for i in range(n_inputs)]
        outputs = [f"{prefix}Y{i}" for i in range(max(1, schema.min_output))]
        # Later nodes share the opset import of their domain.
        imports[schema.domain] = max(
            imports.get(schema.domain, 0), schema.since_version
        )
        if k == 0 and inputs and fdp.ConsumeBool():
            # A custom op can produce unknown types even in a checker-valid model.
            graph_inputs.append(
                helper.make_tensor_value_info("source", TensorProto.FLOAT, [2, 3])
            )
            nodes.append(
                helper.make_node("Custom", ["source"], inputs, domain="fuzz.custom")
            )
            imports["fuzz.custom"] = 1
        else:
            inputs = [
                _schema_input(
                    fdp, schema, i, name, produced, graph_inputs, initializers
                )
                for i, name in enumerate(inputs)
            ]
        node = helper.make_node(schema.name, inputs, outputs, domain=schema.domain)
        node.attribute.extend(_schema_attributes(fdp, schema))
        nodes.append(node)
        produced.extend(outputs)
        if not fdp.ConsumeBool():
            break
    graph = helper.make_graph(
        nodes, "schema_graph", graph_inputs, [], initializer=initializers
    )
    return helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid(d, v) for d, v in imports.items()],
    )


def _const_bool(name, value=True):
    tensor = helper.make_tensor(name, TensorProto.BOOL, [], [value])
    return helper.make_node("Constant", [], [name], value=tensor)


def _build_branch(fdp, depth, max_depth):
    """Build a self-contained subgraph.

    Self-contained means the subgraph produces its own starting tensor via
    a Constant node, so the branch does not depend on outer-scope captures
    we did not declare. With probability the branch nests one of
    If/Loop/Scan, which is what drives the recursion inside shape_inference.
    Loop and Scan body subgraphs are deliberately not signature-conformant;
    the recursive visitor descends before signature checks run, so the
    recursion path is still exercised even when inference ultimately fails.
    """
    nodes = []
    start = f"s_{depth}"
    start_tensor = helper.make_tensor(start, TensorProto.FLOAT, [1], [0.0])
    nodes.append(helper.make_node("Constant", [], [start], value=start_tensor))

    if depth < max_depth and fdp.ConsumeBool():
        sub_op = _SUBGRAPH_OPS[fdp.ConsumeIntInRange(0, len(_SUBGRAPH_OPS) - 1)]
        out = f"sub_{depth}"
        body = _build_branch(fdp, depth + 1, max_depth)
        if sub_op == "If":
            cond = f"c_{depth}"
            nodes.append(_const_bool(cond))
            else_body = _build_branch(fdp, depth + 1, max_depth)
            nodes.append(
                helper.make_node(
                    "If",
                    [cond],
                    [out],
                    then_branch=body,
                    else_branch=else_body,
                )
            )
        elif sub_op == "Loop":
            trip = f"M_{depth}"
            trip_t = helper.make_tensor(trip, TensorProto.INT64, [], [1])
            nodes.append(helper.make_node("Constant", [], [trip], value=trip_t))
            cond = f"c_{depth}"
            nodes.append(_const_bool(cond))
            nodes.append(
                helper.make_node(
                    "Loop",
                    [trip, cond],
                    [out],
                    body=body,
                )
            )
        else:  # Scan
            nodes.append(
                helper.make_node(
                    "Scan",
                    [start],
                    [out],
                    body=body,
                    num_scan_inputs=1,
                )
            )
        last = out
    else:
        last = start
        n_ops = fdp.ConsumeIntInRange(0, 4)
        for i in range(n_ops):
            op = _UNARY[fdp.ConsumeIntInRange(0, len(_UNARY) - 1)]
            nxt = f"v_{depth}_{i}"
            nodes.append(helper.make_node(op, [last], [nxt]))
            last = nxt

    return helper.make_graph(
        nodes,
        f"branch_{depth}",
        inputs=[],
        outputs=[helper.make_tensor_value_info(last, TensorProto.FLOAT, None)],
    )


def _build_model(fdp):
    if fdp.ConsumeBool():
        return _build_schema_model(fdp)
    # Top-level graph mirrors a branch but lives at depth 0 and chooses its
    # own opset version so different shape-inference codepaths (per-opset
    # schemas) are reached.
    max_depth = fdp.ConsumeIntInRange(0, 80)
    graph = _build_branch(fdp, depth=0, max_depth=max_depth)
    opset = fdp.ConsumeIntInRange(7, defs.onnx_opset_version())
    return helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", opset)],
    )


def _known_types(graph):
    types = {}
    for tensor in graph.initializer:
        types[tensor.name] = helper.make_tensor_type_proto(
            tensor.data_type, list(tensor.dims)
        )
    for value in (*graph.input, *graph.value_info, *graph.output):
        if value.HasField("type"):
            types[value.name] = value.type
    return types


def _check_output_types(model):
    """Inferred output types must satisfy the schema when the inputs do."""
    opsets = {
        ("" if opset.domain == "ai.onnx" else opset.domain): opset.version
        for opset in model.opset_import
    }
    types = _known_types(model.graph)
    for node in model.graph.node:
        domain = "" if node.domain == "ai.onnx" else node.domain
        if domain not in opsets or not defs.has(node.op_type, opsets[domain], domain):
            continue
        schema = defs.get_schema(node.op_type, opsets[domain], domain)
        if (not schema.inputs and node.input) or not schema.outputs:
            continue
        bound: dict[str, str] = {}
        inputs_valid = True
        for i, name in enumerate(node.input):
            if not name:
                continue
            param = schema.inputs[min(i, len(schema.inputs) - 1)]
            type_str = _type_to_str(types.get(name))
            if (
                type_str is None
                or (param.types and type_str not in param.types)
                or (
                    param.is_homogeneous
                    and bound.setdefault(param.type_str, type_str) != type_str
                )
            ):
                inputs_valid = False
                break
        if not inputs_valid:
            continue
        for i, name in enumerate(node.output):
            type_str = _type_to_str(types.get(name)) if name else None
            param = schema.outputs[min(i, len(schema.outputs) - 1)]
            # Outputs typed by an attribute, such as Cast's "to", are only
            # validated with check_type and are not bound by the inputs.
            expected = bound.get(param.type_str) if param.is_homogeneous else None
            if type_str is None or expected is None:
                continue
            assert expected == type_str, (
                domain,
                node.op_type,
                schema.since_version,
                param.name,
                type_str,
                expected,
                "Inferred output type violates the type constraint of the inputs",
            )


def _is_repeated(field):
    is_repeated = getattr(field, "is_repeated", None)
    if is_repeated is not None:
        return is_repeated
    return field.label == field.LABEL_REPEATED


def _clear_defaults(message):
    # An explicit zero, such as elem_type UNDEFINED, equals an absent field.
    # Oneof members like dim_value carry meaning even when zero.
    for field, value in message.ListFields():
        if _is_repeated(field):
            if field.message_type is not None:
                for item in value:
                    _clear_defaults(item)
        elif field.message_type is not None:
            _clear_defaults(value)
        elif field.containing_oneof is None and value == field.default_value:
            message.ClearField(field.name)
    return message


def _value_types(graph):
    return {
        value.name: _clear_defaults(
            onnx.TypeProto.FromString(value.type.SerializeToString())
        ).SerializeToString(deterministic=True)
        for value in (*graph.value_info, *graph.output)
        if value.HasField("type")
    }


def _check_inferred(inferred, options):
    """Inference must accept its own output and keep the inferred types."""
    # The type check may flag inputs that only become visible once inferred,
    # so it is left to the output type conformance check below.
    options = {**options, "check_type": False}
    try:
        again = shape_inference.infer_shapes(inferred, **options)
    except Exception as error:
        raise AssertionError("Shape inference rejected its own output") from error
    before = _value_types(inferred.graph)
    after = _value_types(again.graph)
    for name, type_proto in before.items():
        assert after.get(name) == type_proto, (
            name,
            "Inferring an inferred model changed an inferred type",
        )
    _check_output_types(inferred)


def TestOneInput(data):
    # Preserve complete protobuf seeds, regardless of their final byte.
    # The structured flag only applies when the raw model cannot be parsed.
    if len(data) < 2:
        return
    toggles = data[-1]
    options = {
        "strict_mode": bool(toggles & 0x01),
        "check_type": bool(toggles & 0x02),
        "data_prop": bool(toggles & 0x08),
    }
    use_structured = bool(toggles & 0x04)
    # bits 0x10..0x80 are reserved for future toggles; mutations against
    # them are harmless until claimed.

    try:
        try:
            model = onnx.load_model_from_string(data)
        except Exception:
            if not use_structured:
                return
            fdp = atheris.FuzzedDataProvider(data[:-1])
            model = _build_model(fdp)
        inferred = shape_inference.infer_shapes(model, **options)
    except Exception:
        # Malformed fuzz inputs raise a broad set of expected exceptions
        # (ValidationError, InferenceError, DecodeError, ValueError, ...).
        # Real bugs surface as crashes, hangs, sanitizer reports, or
        # inconsistent results of a successful inference.
        return
    _check_inferred(inferred, options)


def main():
    atheris.instrument_all()
    atheris.Setup(sys.argv, TestOneInput, enable_python_coverage=True)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
