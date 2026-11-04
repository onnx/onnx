# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Atheris fuzz harness for onnx.shape_inference.

Two input paths are exercised per iteration, selected by a fuzzer-controlled
toggle byte read from the *tail* of the input (so the head remains a valid
candidate for the raw-bytes path):

* Raw bytes  -> onnx.load_model_from_string  -> infer_shapes
  Catches protobuf parser bugs and bugs reachable only through crafted
  serialized models the structured builder will not produce.

* Structured -> helper.make_model from FuzzedDataProvider -> infer_shapes
  Constructs ModelProto objects whose graphs include subgraph-bearing
  ops (If / Loop / Scan) so the recursive visitor inside shape_inference
  is reached on most iterations rather than only when the parser happens
  to accept a random byte string.

Both strict_mode values and both check_type values are sampled.
"""

from __future__ import annotations

import sys

import atheris

with atheris.instrument_imports():
    import onnx
    from onnx import TensorProto, helper, shape_inference

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
    # Top-level graph mirrors a branch but lives at depth 0 and chooses its
    # own opset version so different shape-inference codepaths (per-opset
    # schemas) are reached.
    max_depth = fdp.ConsumeIntInRange(0, 80)
    graph = _build_branch(fdp, depth=0, max_depth=max_depth)
    opset = fdp.ConsumeIntInRange(7, 27)
    return helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", opset)],
    )


def TestOneInput(data):
    # Toggles live in the trailing byte. On the structured path we slice the
    # byte off before handing the rest to FuzzedDataProvider. On the raw path
    # we pass the full `data` to the protobuf parser unchanged: seed models
    # are complete serialized ModelProtos, so slicing the tail would truncate
    # every seed. The trailing toggle byte becomes part of the raw input,
    # which libFuzzer mutates freely anyway.
    if len(data) < 2:
        return
    toggles = data[-1]
    strict = bool(toggles & 0x01)
    check_type = bool(toggles & 0x02)
    use_structured = bool(toggles & 0x04)
    # bits 0x08..0x80 are reserved for future toggles; mutations against
    # them are harmless until claimed.

    try:
        if use_structured:
            fdp = atheris.FuzzedDataProvider(data[:-1])
            model = _build_model(fdp)
        else:
            model = onnx.load_model_from_string(data)
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
