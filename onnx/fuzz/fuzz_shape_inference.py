# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Atheris fuzz harness for onnx.shape_inference.

Valid serialized models always take the raw path. If parsing fails, a
fuzzer-controlled toggle byte at the tail can select the structured fallback:

* Raw bytes  -> onnx.load_model_from_string  -> infer_shapes
  Catches protobuf parser bugs and bugs reachable only through crafted
  serialized models the structured builder will not produce.

* Structured -> helper.make_model from FuzzedDataProvider -> infer_shapes
  Samples registered operator schemas with unknown input types, or builds
  graphs with subgraph-bearing ops (If / Loop / Scan) to reach the recursive
  visitor without relying on random bytes parsing as a model.

Both strict_mode values and both check_type values are sampled.
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


def _build_unknown_type_model(fdp):
    schema = _SCHEMAS[fdp.ConsumeIntInRange(0, len(_SCHEMAS) - 1)]
    inputs = [f"X{i}" for i in range(schema.min_input)]
    outputs = [f"Y{i}" for i in range(max(1, schema.min_output))]
    # Supply attributes needed to reach the inference code in #8651 rather
    # than stopping at an unrelated missing-attribute error.
    attrs: dict[str, Any] = {}
    if schema.domain == "ai.onnx.ml":
        if schema.name == "DictVectorizer":
            attrs = {"string_vocabulary": ["a"]}
        elif schema.name == "LabelEncoder":
            attrs = (
                {"classes_strings": ["a"]}
                if schema.since_version == 1
                else {"keys_strings": ["a"], "values_int64s": [1]}
            )
        elif schema.name == "TreeEnsemble":
            attrs = {
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
            }
    elif schema.name == "RegexFullMatch":
        attrs = {"pattern": "a.*"}

    nodes = []
    imports = [helper.make_opsetid(schema.domain, schema.since_version)]
    if fdp.ConsumeBool():
        # A custom op can produce unknown types even in a checker-valid model.
        graph_inputs = [
            helper.make_tensor_value_info("source", TensorProto.FLOAT, [2, 3])
        ]
        nodes.append(
            helper.make_node("Custom", ["source"], inputs, domain="fuzz.custom")
        )
        imports.append(helper.make_opsetid("fuzz.custom", 1))
    else:
        graph_inputs = [onnx.ValueInfoProto(name=name) for name in inputs]
    nodes.append(
        helper.make_node(schema.name, inputs, outputs, domain=schema.domain, **attrs)
    )
    graph = helper.make_graph(nodes, "unknown_type", graph_inputs, [])
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
        return _build_unknown_type_model(fdp)
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
    # bits 0x08..0x80 are reserved for future toggles; mutations against
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
