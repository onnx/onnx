# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Atheris fuzz harness for onnx.compose.

Two input paths are exercised per iteration, selected by a fuzzer-controlled
toggle byte read from the *tail* of the input:

* Raw    -> a 4-byte big-endian length prefix splits the remaining bytes
  into m1 | m2, each parsed via onnx.load_model_from_string. Catches
  protobuf parser bugs and bugs reachable only through crafted serialized
  models the structured builder will not produce.

* Structured -> helper.make_model from FuzzedDataProvider builds two small
  linear graphs with predictable input/output names, so merge_models
  reaches its connect_io/add_prefix logic on most iterations rather than
  only when the parser happens to accept a random byte string.

Further bits of the toggle byte select prefix1/prefix2 (add_prefix
collision-resolution path) and a randomized (vs. derived) io_map, so a
single harness covers merge_models, merge_graphs, check_overlapping_names,
the recursive connect_io subgraph rewrite, and add_prefix.
"""

from __future__ import annotations

import random
import struct
import sys

import atheris

with atheris.instrument_imports():
    import onnx
    from onnx import TensorProto, checker, compose, helper

_UNARY = ("Relu", "Sigmoid", "Tanh", "Abs", "Neg", "Identity")


def _value_info(name):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, ["N"])


def _build_model(fdp, tag):
    """Build a small linear graph with predictable input/output names.

    Naming the sole input/output `In<tag>`/`Out<tag>` lets the derived
    io_map (m1's outputs -> m2's inputs) connect on most iterations, so
    merge_models' rewrite logic is reached without relying on the raw
    protobuf parser to produce compatible names.
    """
    n_ops = fdp.ConsumeIntInRange(0, 4)
    last = f"In{tag}"
    nodes = []
    for i in range(n_ops):
        op = _UNARY[fdp.ConsumeIntInRange(0, len(_UNARY) - 1)]
        nxt = f"v{tag}_{i}"
        nodes.append(helper.make_node(op, [last], [nxt]))
        last = nxt
    out_name = f"Out{tag}"
    nodes.append(helper.make_node("Identity", [last], [out_name]))
    graph = helper.make_graph(
        nodes, f"g{tag}", [_value_info(f"In{tag}")], [_value_info(out_name)]
    )
    opset = fdp.ConsumeIntInRange(7, onnx.defs.onnx_opset_version())
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])


def _derived_io_map(m1, m2):
    outs = [o.name for o in m1.graph.output]
    ins = [i.name for i in m2.graph.input]
    return list(zip(outs, ins, strict=False))


def _random_io_map(data, m1, m2):
    outs = [o.name for o in m1.graph.output]
    ins = [i.name for i in m2.graph.input]
    if not outs or not ins:
        return []
    rng = random.Random(data)
    rng.shuffle(outs)
    rng.shuffle(ins)
    n = rng.randint(0, min(len(outs), len(ins)))
    return list(zip(outs[:n], ins[:n], strict=True))


def TestOneInput(data):
    # Toggles live in the trailing byte; see module docstring and
    # onnx/fuzz/README.md's "fuzz_compose.py toggle byte" table.
    if len(data) < 2:
        return
    toggles = data[-1]
    body = data[:-1]
    use_prefix = bool(toggles & 0x01)
    use_structured = bool(toggles & 0x04)
    use_random_io_map = bool(toggles & 0x08)
    # bits 0x02, 0x10..0x80 are reserved for future toggles; mutations
    # against them are harmless until claimed.

    try:
        if use_structured:
            fdp = atheris.FuzzedDataProvider(body)
            m1 = _build_model(fdp, 1)
            m2 = _build_model(fdp, 2)
        else:
            if len(body) < 4:
                return
            (n1,) = struct.unpack(">I", body[:4])
            rest = body[4:]
            if n1 > len(rest):
                return
            m1 = onnx.load_model_from_string(rest[:n1])
            m2 = onnx.load_model_from_string(rest[n1:])

        io_map = (
            _random_io_map(body, m1, m2)
            if use_random_io_map
            else _derived_io_map(m1, m2)
        )

        kwargs = {}
        if use_prefix:
            kwargs["prefix1"] = "g1_"
            kwargs["prefix2"] = "g2_"

        merged = compose.merge_models(m1, m2, io_map, **kwargs)
        checker.check_model(merged, full_check=True)
    except Exception:
        # Malformed fuzz inputs raise a broad set of expected exceptions
        # (ValidationError, TypeError, ValueError, DecodeError, ...).
        # Real bugs surface as crashes, hangs, or sanitizer reports.
        return


def main():
    atheris.instrument_all()
    atheris.Setup(sys.argv, TestOneInput, enable_python_coverage=True)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
