# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Atheris fuzz harness for onnx.shape_inference.

Valid serialized models always take the raw path. If parsing fails, a
fuzzer-controlled toggle byte at the tail can select the structured fallback:

* Raw bytes  -> onnx.load_model_from_string  -> infer_shapes
  Catches protobuf parser bugs and bugs reachable only through crafted
  serialized models the structured builder will not produce.

* Structured -> helper.make_model from FuzzedDataProvider -> infer_shapes
  Samples registered operator schemas with untyped, typed, constant, or
  omitted inputs and generated attributes, or builds graphs with
  subgraph-bearing ops (If / Loop / Scan) to reach the recursive visitor
  without relying on random bytes parsing as a model.

All combinations of strict_mode, check_type, and data_prop are sampled.
"""

from __future__ import annotations

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
    and schema.min_input > 0
    and schema.has_type_and_shape_inference_function
)


_ELEM_TYPES = tuple(
    value for value in TensorProto.DataType.values() if value != TensorProto.UNDEFINED
)

# Kinds of producer for each schema input.
_UNTYPED, _TYPED, _INITIALIZER, _OMITTED = range(4)
_MAX_EXTRA_INPUTS = 2
_MAX_RANK = 4


def _make_tensor_type(fdp):
    elem_type = _ELEM_TYPES[fdp.ConsumeIntInRange(0, len(_ELEM_TYPES) - 1)]
    rank = fdp.ConsumeIntInRange(-1, _MAX_RANK)
    if rank < 0:
        # No shape information at all.
        return helper.make_tensor_type_proto(elem_type, None)
    shape: list[int | str | None] = []
    for i in range(rank):
        kind = fdp.ConsumeIntInRange(0, 2)
        if kind == 0:
            shape.append(fdp.ConsumeIntInRange(0, 8))
        elif kind == 1:
            shape.append(f"d{i}")
        else:
            shape.append(None)
    return helper.make_tensor_type_proto(elem_type, shape)


def _make_type(fdp):
    kind = fdp.ConsumeIntInRange(0, 7)
    if kind == 6:
        return helper.make_sequence_type_proto(_make_tensor_type(fdp))
    if kind == 7:
        key_type = (TensorProto.STRING, TensorProto.INT64)[fdp.ConsumeIntInRange(0, 1)]
        return helper.make_map_type_proto(key_type, _make_tensor_type(fdp))
    return _make_tensor_type(fdp)


def _make_initializer(fdp, name):
    # Small constant INT64 vectors drive data-dependent inference (Reshape,
    # Expand, Slice, ...); float scalars cover the remaining constant inputs.
    if fdp.ConsumeBool():
        values = [
            fdp.ConsumeIntInRange(-2, 6) for _ in range(fdp.ConsumeIntInRange(0, 4))
        ]
        return helper.make_tensor(name, TensorProto.INT64, [len(values)], values)
    return helper.make_tensor(name, TensorProto.FLOAT, [], [1.0])


def _small_ints(fdp):
    return [fdp.ConsumeIntInRange(-2, 4) for _ in range(fdp.ConsumeIntInRange(0, 3))]


def _make_attribute(fdp, name, attr_type):
    t = defs.OpSchema.AttrType
    if attr_type == t.INT:
        value: Any = fdp.ConsumeIntInRange(-2, 4)
    elif attr_type == t.INTS:
        value = _small_ints(fdp)
    elif attr_type == t.FLOAT:
        value = float(fdp.ConsumeIntInRange(-2, 4))
    elif attr_type == t.FLOATS:
        value = [float(v) for v in _small_ints(fdp)]
    elif attr_type == t.STRING:
        value = "a"
    elif attr_type == t.STRINGS:
        value = ["a"] * fdp.ConsumeIntInRange(0, 3)
    elif attr_type == t.TENSOR:
        value = helper.make_tensor(name, TensorProto.FLOAT, [1], [1.0])
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
            proto = _make_attribute(fdp, name, attr.type)
            if proto is not None:
                attrs[name] = proto
    for name, value in _fixed_attributes(schema).items():
        attrs[name] = helper.make_attribute(name, value)
    return list(attrs.values())


def _build_schema_model(fdp):
    schema = _SCHEMAS[fdp.ConsumeIntInRange(0, len(_SCHEMAS) - 1)]
    n_inputs = fdp.ConsumeIntInRange(
        schema.min_input,
        min(schema.max_input, schema.min_input + _MAX_EXTRA_INPUTS),
    )
    inputs = [f"X{i}" for i in range(n_inputs)]
    outputs = [f"Y{i}" for i in range(max(1, schema.min_output))]

    nodes = []
    imports = [helper.make_opsetid(schema.domain, schema.since_version)]
    graph_inputs = []
    initializers = []
    if fdp.ConsumeBool():
        # A custom op can produce unknown types even in a checker-valid model.
        graph_inputs.append(
            helper.make_tensor_value_info("source", TensorProto.FLOAT, [2, 3])
        )
        nodes.append(
            helper.make_node("Custom", ["source"], inputs, domain="fuzz.custom")
        )
        imports.append(helper.make_opsetid("fuzz.custom", 1))
    else:
        for i, name in enumerate(inputs):
            kind = fdp.ConsumeIntInRange(_UNTYPED, _OMITTED)
            if kind == _UNTYPED:
                graph_inputs.append(onnx.ValueInfoProto(name=name))
            elif kind == _TYPED:
                graph_inputs.append(helper.make_value_info(name, _make_type(fdp)))
            elif kind == _INITIALIZER:
                initializers.append(_make_initializer(fdp, name))
            else:
                # An empty name marks an omitted optional input.
                inputs[i] = ""
    node = helper.make_node(schema.name, inputs, outputs, domain=schema.domain)
    node.attribute.extend(_schema_attributes(fdp, schema))
    nodes.append(node)
    graph = helper.make_graph(
        nodes, "schema_graph", graph_inputs, [], initializer=initializers
    )
    return helper.make_model(graph, opset_imports=imports)


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


def TestOneInput(data):
    # Preserve complete protobuf seeds, regardless of their final byte.
    # The structured flag only applies when the raw model cannot be parsed.
    if len(data) < 2:
        return
    toggles = data[-1]
    strict = bool(toggles & 0x01)
    check_type = bool(toggles & 0x02)
    use_structured = bool(toggles & 0x04)
    data_prop = bool(toggles & 0x08)
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
        shape_inference.infer_shapes(
            model,
            check_type=check_type,
            strict_mode=strict,
            data_prop=data_prop,
        )
    except Exception:
        # Malformed fuzz inputs raise a broad set of expected exceptions
        # (ValidationError, InferenceError, DecodeError, ValueError, ...).
        # Real bugs surface as crashes, hangs, or sanitizer reports.
        return


def main():
    atheris.instrument_all()
    atheris.Setup(sys.argv, TestOneInput, enable_python_coverage=True)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
