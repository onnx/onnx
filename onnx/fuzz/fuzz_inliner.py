# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Atheris fuzz harness for onnx.inliner.

The inliner (onnx/inliner/inliner.cc) rewrites call-sites of model-local
FunctionProtos (and, optionally, schema-defined function ops) into the
calling graph. It renames every intermediate value, substitutes attribute
references (ref_attr_name) and attribute defaults, recurses into nested
calls and control-flow subgraphs, and can version-convert function bodies
whose opset differs from the model's. None of this was reached by the
existing harnesses.

Two input paths are selected by a toggle byte read from the *tail* of the
input:

* Raw        -> onnx.load_model_from_string. Reaches inliner paths that only
  crafted serialized models trigger (dangling names, odd domains, malformed
  attributes, duplicate functions).

* Structured -> FuzzedDataProvider builds a model with 1-4 local functions
  that may call each other, use attribute references with or without
  defaults, contain If subgraphs, and declare an opset different from the
  model's. This reaches the renamer and attribute-binding logic on most
  iterations instead of only when random bytes happen to parse.

Toggle byte bits:

  0x01  structured (vs raw) input
  0x02  convert_version=True for inline_local_functions
  0x04  use inline_selected_functions instead of inline_local_functions
  0x08  exclude=True for inline_selected_functions
  0x10  inline schema-defined functions as well (inline_selected_functions2)
  0x20  structured only: allow a function to call itself or a later function
        (call cycles). Guards the inliner against unbounded recursion.
  0x40  run checker.check_model on the inlined model
  0x80  reserved

Oracle (structured path, acyclic, matching opsets, inline_local_functions):
after inlining, no node anywhere in the main graph -- including nested
subgraphs -- may still call a model-local function. A remaining call means
the inliner silently skipped a call-site.
"""

from __future__ import annotations

import sys

import atheris

with atheris.instrument_imports():
    from onnx import (
        AttributeProto,
        TensorProto,
        checker,
        defs,
        helper,
        inliner,
        load_model_from_string,
    )

_LOCAL_DOMAINS = ("local", "local.b")
_UNARY = ("Relu", "Sigmoid", "Tanh", "Abs", "Neg", "Identity")
_BINARY = ("Add", "Mul", "Sub")
# Standard ops that are defined by a FunctionProto in the schema registry, so
# inline_selected_functions2 has something to expand.
_SCHEMA_FUNCTION_OPS = ("Celu", "Selu", "HardSwish", "Softsign", "Mish")
_MIN_OPSET = 18  # attribute_proto defaults and the ops above are all valid


def _vi(name):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, ["N"])


def _attr_ref(name, ref, attr_type=AttributeProto.FLOAT):
    """An attribute that refers to the enclosing function's attribute `ref`."""
    attr = AttributeProto()
    attr.name = name
    attr.ref_attr_name = ref
    attr.type = attr_type
    return attr


def _function_body(fdp, idx, n_funcs, allow_cycles, fn_names, depth=0):
    """Return (nodes, output_name) for a linear chain starting at input 'x'."""
    nodes = []
    last = "x"
    for step in range(fdp.ConsumeIntInRange(1, 4)):
        nxt = f"t{depth}_{step}"
        choice = fdp.ConsumeIntInRange(0, 5)
        if choice == 0:
            op = _UNARY[fdp.ConsumeIntInRange(0, len(_UNARY) - 1)]
            nodes.append(helper.make_node(op, [last], [nxt]))
        elif choice == 1:
            # Constant whose value comes from the function attribute 'alpha'.
            # Exercises ref_attr_name substitution and attribute defaults.
            c = f"c{depth}_{step}"
            const = helper.make_node("Constant", [], [c])
            const.attribute.append(_attr_ref("value_float", "alpha"))
            op = _BINARY[fdp.ConsumeIntInRange(0, len(_BINARY) - 1)]
            nodes.extend([const, helper.make_node(op, [last, c], [nxt])])
        elif choice == 2 and n_funcs > 1:
            # Call another local function, forwarding (or overriding) alpha.
            if allow_cycles:
                callee = fdp.ConsumeIntInRange(0, n_funcs - 1)
            elif idx > 0:
                callee = fdp.ConsumeIntInRange(0, idx - 1)
            else:
                callee = None
            if callee is None:
                nodes.append(helper.make_node("Identity", [last], [nxt]))
            else:
                domain, name = fn_names[callee]
                call = helper.make_node(name, [last], [nxt], domain=domain)
                if fdp.ConsumeBool():
                    call.attribute.append(_attr_ref("alpha", "alpha"))
                elif fdp.ConsumeBool():
                    call.attribute.append(
                        helper.make_attribute("alpha", fdp.ConsumeFloat())
                    )
                nodes.append(call)
        elif choice == 3 and depth < 2:
            # If node whose branches are subgraphs; the inliner must rename
            # values captured from the outer scope consistently.
            cond = f"cond{depth}_{step}"
            nodes.append(
                helper.make_node(
                    "Constant",
                    [],
                    [cond],
                    value=helper.make_tensor("v", TensorProto.BOOL, [], [True]),
                )
            )
            branches = []
            for tag in ("then", "else"):
                out = f"{tag}{depth}_{step}"
                op = _UNARY[fdp.ConsumeIntInRange(0, len(_UNARY) - 1)]
                body = [helper.make_node(op, [last], [out])]  # outer-scope read
                branches.append(
                    helper.make_graph(body, f"{tag}_{idx}_{depth}", [], [_vi(out)])
                )
            nodes.append(
                helper.make_node(
                    "If",
                    [cond],
                    [nxt],
                    then_branch=branches[0],
                    else_branch=branches[1],
                )
            )
        else:
            nodes.append(helper.make_node("Identity", [last], [nxt]))
        last = nxt
    return nodes, last


def _build_model(fdp, allow_cycles):
    latest = defs.onnx_opset_version()
    model_opset = fdp.ConsumeIntInRange(_MIN_OPSET, latest)
    n_funcs = fdp.ConsumeIntInRange(1, 4)
    fn_names = [
        (_LOCAL_DOMAINS[fdp.ConsumeIntInRange(0, 1)], f"F{i}") for i in range(n_funcs)
    ]

    functions = []
    opsets_match = True
    for i, (domain, name) in enumerate(fn_names):
        nodes, out = _function_body(fdp, i, n_funcs, allow_cycles, fn_names)
        nodes.append(helper.make_node("Identity", [out], ["y"]))
        fn_opset = model_opset
        if fdp.ConsumeIntInRange(0, 3) == 0:
            fn_opset = fdp.ConsumeIntInRange(_MIN_OPSET, latest)
        opsets_match &= fn_opset == model_opset
        fn = helper.make_function(
            domain,
            name,
            ["x"],
            ["y"],
            nodes,
            [helper.make_opsetid("", fn_opset)]
            + [helper.make_opsetid(d, 1) for d in _LOCAL_DOMAINS],
            attributes=[] if fdp.ConsumeBool() else ["alpha"],
        )
        if fdp.ConsumeBool():
            # Default value for alpha via attribute_proto (onnx >= 1.15).
            del fn.attribute[:]
            fn.attribute_proto.append(
                helper.make_attribute("alpha", fdp.ConsumeFloat())
            )
        functions.append(fn)

    # Main graph: a chain of local-function calls and schema-function ops,
    # with an occasional call nested inside an If branch.
    nodes = []
    last = "X"
    for step in range(fdp.ConsumeIntInRange(1, 5)):
        nxt = f"m{step}"
        kind = fdp.ConsumeIntInRange(0, 3)
        if kind <= 1:
            domain, name = fn_names[fdp.ConsumeIntInRange(0, n_funcs - 1)]
            call = helper.make_node(name, [last], [nxt], domain=domain)
            if fdp.ConsumeBool():
                call.attribute.append(
                    helper.make_attribute("alpha", fdp.ConsumeFloat())
                )
            nodes.append(call)
        elif kind == 2:
            op = _SCHEMA_FUNCTION_OPS[
                fdp.ConsumeIntInRange(0, len(_SCHEMA_FUNCTION_OPS) - 1)
            ]
            nodes.append(helper.make_node(op, [last], [nxt]))
        else:
            domain, name = fn_names[fdp.ConsumeIntInRange(0, n_funcs - 1)]
            cond = f"mc{step}"
            nodes.append(
                helper.make_node(
                    "Constant",
                    [],
                    [cond],
                    value=helper.make_tensor("v", TensorProto.BOOL, [], [True]),
                )
            )
            branches = []
            for tag in ("then", "else"):
                out = f"m{tag}{step}"
                body = [helper.make_node(name, [last], [out], domain=domain)]
                branches.append(
                    helper.make_graph(body, f"m{tag}{step}", [], [_vi(out)])
                )
            nodes.append(
                helper.make_node(
                    "If",
                    [cond],
                    [nxt],
                    then_branch=branches[0],
                    else_branch=branches[1],
                )
            )
        last = nxt
    nodes.append(helper.make_node("Identity", [last], ["Y"]))

    graph = helper.make_graph(nodes, "main", [_vi("X")], [_vi("Y")])
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", model_opset)]
        + [helper.make_opsetid(d, 1) for d in _LOCAL_DOMAINS],
        functions=functions,
    )
    return model, fn_names, opsets_match


def _iter_nodes(graph):
    for node in graph.node:
        yield node
        for attr in node.attribute:
            if attr.type == AttributeProto.GRAPH:
                yield from _iter_nodes(attr.g)
            elif attr.type == AttributeProto.GRAPHS:
                for g in attr.graphs:
                    yield from _iter_nodes(g)


def TestOneInput(data):
    if len(data) < 2:
        return
    toggles = data[-1]
    body = data[:-1]
    structured = bool(toggles & 0x01)
    convert_version = bool(toggles & 0x02)
    use_selected = bool(toggles & 0x04)
    exclude = bool(toggles & 0x08)
    schema_functions = bool(toggles & 0x10)
    allow_cycles = bool(toggles & 0x20)
    run_checker = bool(toggles & 0x40)

    fn_names = []
    oracle = False
    try:
        if structured:
            fdp = atheris.FuzzedDataProvider(body)
            model, fn_names, opsets_match = _build_model(fdp, allow_cycles)
            oracle = opsets_match and not allow_cycles and not use_selected
        else:
            model = load_model_from_string(body)
            fn_names = [(f.domain, f.name) for f in model.functions]

        if use_selected:
            # Select every other local function so both the "inline" and
            # "keep" branches of FunctionIdSet are exercised. When schema
            # functions are enabled, select some of them too: membership is
            # checked before expansion, so include mode would otherwise never
            # expand a schema function.
            ids = fn_names[::2]
            if schema_functions:
                ids = ids + [("", op) for op in _SCHEMA_FUNCTION_OPS[::2]]
            result = inliner.inline_selected_functions(
                model, ids, exclude=exclude, inline_schema_functions=schema_functions
            )
        else:
            result = inliner.inline_local_functions(model, convert_version)
    except Exception:
        # Malformed inputs raise expected errors (DecodeError,
        # ValidationError, ConvertError, RuntimeError, ...). Real bugs
        # surface as crashes, hangs, sanitizer reports, or the oracle below.
        return

    # Run the oracle before the checker so a validation failure on the
    # inlined model cannot hide a skipped call-site.
    if oracle:
        local = set(fn_names)
        for node in _iter_nodes(result.graph):
            if (node.domain, node.op_type) in local:
                raise AssertionError(
                    f"inline_local_functions left a call to "
                    f"{node.domain}::{node.op_type} in the inlined graph"
                )

    if run_checker:
        try:
            checker.check_model(result)
        except Exception:
            return


def main():
    atheris.instrument_all()
    atheris.Setup(sys.argv, TestOneInput, enable_python_coverage=True)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
