# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Atheris fuzz harness for onnx.compose.merge_models.

merge_models is the integration entry point for graph composition: it
transitively exercises merge_graphs, check_overlapping_names, the
recursive connect_io subgraph rewrite, add_prefix(_graph), and a final
checker.check_model on the merged result.

The harness drives two ModelProtos plus an io_map into merge_models.
A trailing toggle byte selects the input strategy:

    bit 0x01  pass prefix1="p1/" / prefix2="p2/"  (collision-resolution path)
    bit 0x02  reserved
    bit 0x04  structured: build both models from FuzzedDataProvider
              (else raw: a 4-byte big-endian length prefix splits the
               remaining bytes into m1 | m2 serialized ModelProtos)
    bit 0x08  use a random io_map (else derive it from the actual
              output names of m1 and input names of m2)
    bits 0x10..0x80 reserved

Seeds (see make_seed_corpus.py) target the raw path with a derived
io_map (toggle 0x00 / 0x01), so each seed reaches merge logic on the
first iteration instead of dying at the protobuf parser.
"""

from __future__ import annotations

import struct
import sys

import atheris

with atheris.instrument_imports():
    import onnx
    from onnx import TensorProto, compose, helper

_TOG_PREFIX = 0x01
_TOG_STRUCTURED = 0x04
_TOG_RANDOM_IOMAP = 0x08

# Kept well below CPython's default recursion limit (~1000). libFuzzer
# mutation can still push past this; RecursionError is caught as an
# expected (recoverable) Python error rather than treated as a finding.
_MAX_IF_DEPTH = 30

_BINARY_OPS = ("Add", "Sub", "Mul")


def _value_info(name: str):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, ["N", "M"])


def _nested_if_chain(src: str, depth: int, tag: str):
    """Build a `depth`-nested chain of If nodes whose branches reference
    `src` from the enclosing scope. Returns (nodes, output_name).

    Each level wraps the previous level inside its then_branch, so a
    merge that rewires `src` forces connect_io into its recursive
    AttributeProto.GRAPH descent at every level.
    """
    inner_nodes = [helper.make_node("Relu", [src], [f"{tag}_inner"])]
    inner_out = f"{tag}_inner"

    for d in range(depth):
        cond = f"{tag}_cond{d}"
        cond_node = helper.make_node(
            "Constant",
            [],
            [cond],
            value=helper.make_tensor(f"{tag}_cv{d}", TensorProto.BOOL, [], [1]),
        )
        then_g = helper.make_graph(
            inner_nodes,
            f"{tag}_then{d}",
            [],
            [helper.make_tensor_value_info(inner_out, TensorProto.FLOAT, None)],
        )
        else_out = f"{tag}_else{d}"
        else_g = helper.make_graph(
            [helper.make_node("Identity", [src], [else_out])],
            f"{tag}_elseg{d}",
            [],
            [helper.make_tensor_value_info(else_out, TensorProto.FLOAT, None)],
        )
        if_out = f"{tag}_if{d}"
        if_node = helper.make_node(
            "If", [cond], [if_out], then_branch=then_g, else_branch=else_g
        )
        inner_nodes = [cond_node, if_node]
        inner_out = if_out

    return inner_nodes, inner_out


def _build_model(fdp: atheris.FuzzedDataProvider, tag: str):
    """Build a small valid single-input model.

    Returns (ModelProto, input_names, output_names). With ~50% probability
    the body is a nested-If chain (exercising subgraph recursion); otherwise
    it is a flat set of elementwise binary nodes producing 1-3 outputs.
    """
    src = f"{tag}_X"

    if fdp.ConsumeBool():
        depth = fdp.ConsumeIntInRange(0, _MAX_IF_DEPTH)
        nodes, out = _nested_if_chain(src, depth, tag)
        outputs = [out]
    else:
        nodes = []
        outputs = []
        n_out = fdp.ConsumeIntInRange(1, 3)
        for i in range(n_out):
            op = fdp.PickValueInList(_BINARY_OPS)
            out = f"{tag}_o{i}"
            nodes.append(helper.make_node(op, [src, src], [out]))
            outputs.append(out)

    graph = helper.make_graph(
        nodes,
        f"{tag}_g",
        [_value_info(src)],
        [_value_info(o) for o in outputs],
    )
    opset = fdp.ConsumeIntInRange(11, 20)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    return model, [src], outputs


def _make_io_map(fdp, toggles, m1_out, m2_in):
    """Derive an io_map from real names (high merge-logic coverage) or, when
    requested or when either side has no names, emit a random one (stresses
    io_map / collision validation error paths)."""
    if (toggles & _TOG_RANDOM_IOMAP) or not (m1_out and m2_in):
        pairs = []
        for _ in range(fdp.ConsumeIntInRange(0, 4)):
            a = fdp.ConsumeUnicodeNoSurrogates(8) or "a"
            b = fdp.ConsumeUnicodeNoSurrogates(8) or "b"
            pairs.append((a, b))
        return pairs

    k = min(len(m1_out), len(m2_in))
    k = fdp.ConsumeIntInRange(0, k)
    return [(m1_out[i], m2_in[i]) for i in range(k)]


def TestOneInput(data: bytes) -> None:
    if len(data) < 2:
        return

    toggles = data[-1]
    body = data[:-1]
    fdp = atheris.FuzzedDataProvider(body)

    if toggles & _TOG_STRUCTURED:
        try:
            m1, _m1_in, m1_out = _build_model(fdp, "m1")
            m2, m2_in, _m2_out = _build_model(fdp, "m2")
        except Exception:
            return
    else:
        if len(body) <= 4:
            return
        (m1_len,) = struct.unpack(">I", body[:4])
        # Clamp into the payload length (len(body) - 4) so libFuzzer mutations
        # of the length field still split into a non-empty m1 and m2 instead of
        # an empty m2 on every iteration.
        m1_len %= len(body) - 4
        m1_bytes = body[4 : 4 + m1_len]
        m2_bytes = body[4 + m1_len :]
        try:
            m1 = onnx.load_model_from_string(m1_bytes)
            m2 = onnx.load_model_from_string(m2_bytes)
        except Exception:
            return
        m1_out = [o.name for o in m1.graph.output]
        m2_in = [i.name for i in m2.graph.input]

    io_map = _make_io_map(fdp, toggles, m1_out, m2_in)

    kwargs = {}
    if toggles & _TOG_PREFIX:
        kwargs["prefix1"] = "p1/"
        kwargs["prefix2"] = "p2/"

    try:
        compose.merge_models(m1, m2, io_map=io_map, **kwargs)
    except Exception:
        # Expected: ValidationError, InferenceError, DecodeError, ValueError,
        # KeyError, IndexError, RecursionError on hostile input. Only crashes
        # and sanitizer reports are real findings.
        return


def main() -> None:
    atheris.instrument_all()
    atheris.Setup(sys.argv, TestOneInput, enable_python_coverage=True)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
