# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import struct
import sys
import zipfile
from typing import TYPE_CHECKING

from onnx import TensorProto, helper

if TYPE_CHECKING:
    from collections.abc import Mapping


def _make_model(
    op_type: str, opset_version: int, inputs: list[str], attrs=None
) -> bytes:
    if attrs is None:
        attrs = {}

    graph_inputs = []
    if "X" in inputs:
        graph_inputs.append(helper.make_tensor_value_info("X", TensorProto.FLOAT, [1]))
    if "scales" in inputs:
        graph_inputs.append(
            helper.make_tensor_value_info("scales", TensorProto.FLOAT, [1])
        )

    graph_outputs = [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1])]
    node = helper.make_node(op_type, inputs, ["Y"], **attrs)
    graph = helper.make_graph(
        [node], f"{op_type.lower()}-seed", graph_inputs, graph_outputs
    )
    model = helper.make_model(
        graph,
        producer_name="oss-fuzz",
        opset_imports=[helper.make_opsetid("", opset_version)],
    )
    return model.SerializeToString()


# Text-format seeds for fuzz_parser, extracted from onnx/test/parser_test.py.
# Each string is a valid input to onnx.parser.parse_model().
_PARSER_SEEDS: dict[str, str] = {
    # Minimal 3-op linear model; exercises basic node and graph parsing.
    "basic_matmul_softmax.txt": """\

  ir_version: 7,
  opset_import: ["" : 10]
>
agraph (float[N, 128] X, float[128, 10] W, float[10] B) => (float[N] C)
{
   T = MatMul(X, W)
   S = Add(T, B)
   C = Softmax(S)
}
""",
    # Multiple opset imports; exercises opset_import list parsing.
    "multi_opset.txt": """\

  ir_version: 7,
  opset_import: ["" : 10, "com.microsoft" : 1]
>
agraph (float[N, 128] X, float[128, 10] W, float[10] B) => (float[N] C)
{
   T = MatMul(X, W)
   S = Add(T, B)
   C = Softmax(S)
}
""",
    # All top-level metadata fields; exercises producer_name, doc_string, etc.
    "model_with_metadata.txt": """\

  ir_version: 9,
  opset_import: ["" : 15],
  producer_name: "oss-fuzz-seed",
  producer_version: "1.0",
  model_version: 1,
  doc_string: "seed model for fuzz_parser"
>
agraph (float[N] x) => (float[N] y)
{
   y = Relu(x)
}
""",
    # Model with a local function definition and attribute references;
    # exercises the function-proto and attribute-default parsing paths.
    "function_with_attributes.txt": """\

  ir_version: 9,
  opset_import: ["" : 15, "custom_domain" : 1],
  producer_name: "oss-fuzz-seed",
  producer_version: "1.0",
  model_version: 1,
  doc_string: "model with local function"
>
agraph (float[N] x) => (float[N] out)
{
   out = custom_domain.Selu<alpha=2.0, gamma=3.0>(x)
}

  domain: "custom_domain",
  opset_import: ["" : 15],
  doc_string: "custom Selu function"
>
Selu
<alpha: float=1.6732631921768188, gamma: float=1.0507010221481323>
(X) => (C)
{
    constant_alpha = Constant<value_float: float=@alpha>()
    constant_gamma = Constant<value_float: float=@gamma>()
    alpha_x = CastLike(constant_alpha, X)
    gamma_x = CastLike(constant_gamma, X)
    exp_x = Exp(X)
    alpha_x_exp_x = Mul(alpha_x, exp_x)
    alpha_x_exp_x_ = Sub(alpha_x_exp_x, alpha_x)
    neg = Mul(gamma_x, alpha_x_exp_x_)
    pos = Mul(gamma_x, X)
    _zero = Constant<value_float=0.0>()
    zero = CastLike(_zero, X)
    less_eq = LessOrEqual(X, zero)
    C = Where(less_eq, neg, pos)
}
""",
    # Cast op with a type initializer; exercises initializer and attribute parsing.
    "cast_with_initializer.txt": """\

  ir_version: 10,
  opset_import: ["" : 19]
>
agraph (float[N] X) => (int64[N] C)

  int64[1] weight = {0}
>
{
   C = Cast<to=7>(X)
}
""",
    # Special float literal values (inf, -inf, nan); exercises the float
    # literal parser branches that differ from ordinary decimal parsing.
    "float_special_values.txt": """\

  ir_version: 8,
  opset_import: ["" : 18]
>
agraph (float[1] X) => (float[1] Y)
{
    pos_inf = Constant<value_float=inf>()
    neg_inf = Constant<value_float=-inf>()
    not_a_num = Constant<value_float=nan>()
    Y = Add(X, pos_inf)
}
""",
}


# Seed models for fuzz_shape_inference. Each seed is a complete serialized
# ModelProto with no trailing bytes: the harness's raw path passes the full
# input to onnx.load_model_from_string, which rejects any trailing bytes, so
# the toggle byte comes from the model bytes themselves (libFuzzer mutates
# it freely). When the toggle's structured bit is clear, the seed loads and
# reaches infer_shapes directly.


def _si_linear() -> bytes:
    """Linear chain Relu -> Sigmoid; exercises unary shape pass-through."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])
    graph = helper.make_graph(
        [
            helper.make_node("Relu", ["X"], ["T"]),
            helper.make_node("Sigmoid", ["T"], ["Y"]),
        ],
        "linear",
        [X],
        [Y],
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 15)]
    ).SerializeToString()


def _si_concat() -> bytes:
    """Concat on axis 0; exercises variadic-input shape-data propagation."""
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, 4])
    B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [3, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [5, 4])
    graph = helper.make_graph(
        [helper.make_node("Concat", ["A", "B"], ["Y"], axis=0)],
        "concat",
        [A, B],
        [Y],
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 15)]
    ).SerializeToString()


def _si_matmul() -> bytes:
    """MatMul [4,8] x [8,2] -> [4,2]; exercises 2-D shape propagation."""
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [4, 8])
    B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [8, 2])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4, 2])
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["A", "B"], ["Y"])],
        "matmul",
        [A, B],
        [Y],
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 15)]
    ).SerializeToString()


def _si_reshape() -> bytes:
    """Reshape driven by a Constant shape tensor; exercises shape-data propagation."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [8, 1])
    shape_tensor = helper.make_tensor("shape", TensorProto.INT64, [2], [8, 1])
    graph = helper.make_graph(
        [
            helper.make_node("Constant", [], ["shape_out"], value=shape_tensor),
            helper.make_node("Reshape", ["X", "shape_out"], ["Y"]),
        ],
        "reshape",
        [X],
        [Y],
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 15)]
    ).SerializeToString()


def _si_if() -> bytes:
    """If op with then/else branches; exercises subgraph-recursive shape inference."""
    cond_t = helper.make_tensor("cv", TensorProto.BOOL, [], [True])
    cond_node = helper.make_node("Constant", [], ["cond"], value=cond_t)

    def _branch(name: str, value: float):
        t = helper.make_tensor(name, TensorProto.FLOAT, [1], [value])
        return helper.make_graph(
            [helper.make_node("Constant", [], [name], value=t)],
            f"{name}_graph",
            [],
            [helper.make_tensor_value_info(name, TensorProto.FLOAT, [1])],
        )

    if_node = helper.make_node(
        "If",
        ["cond"],
        ["result"],
        then_branch=_branch("then_out", 1.0),
        else_branch=_branch("else_out", 0.0),
    )
    graph = helper.make_graph(
        [cond_node, if_node],
        "if_graph",
        [],
        [helper.make_tensor_value_info("result", TensorProto.FLOAT, [1])],
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 15)]
    ).SerializeToString()


def _si_loop() -> bytes:
    """Loop op with scan output; exercises Loop-subgraph recursive descent."""
    trip_t = helper.make_tensor("trip", TensorProto.INT64, [], [3])
    trip_node = helper.make_node("Constant", [], ["trip_count"], value=trip_t)
    cond_t = helper.make_tensor("ci", TensorProto.BOOL, [], [True])
    cond_node = helper.make_node("Constant", [], ["cond"], value=cond_t)

    body_cond_t = helper.make_tensor("bc", TensorProto.BOOL, [], [True])
    scan_t = helper.make_tensor("sv", TensorProto.FLOAT, [1], [1.0])
    body = helper.make_graph(
        [
            helper.make_node("Constant", [], ["cond_out"], value=body_cond_t),
            helper.make_node("Constant", [], ["scan_out"], value=scan_t),
        ],
        "loop_body",
        [
            helper.make_tensor_value_info("iter_count", TensorProto.INT64, []),
            helper.make_tensor_value_info("cond_in", TensorProto.BOOL, []),
        ],
        [
            helper.make_tensor_value_info("cond_out", TensorProto.BOOL, []),
            helper.make_tensor_value_info("scan_out", TensorProto.FLOAT, [1]),
        ],
    )
    loop_node = helper.make_node(
        "Loop", ["trip_count", "cond"], ["loop_out"], body=body
    )
    graph = helper.make_graph(
        [trip_node, cond_node, loop_node],
        "loop_graph",
        [],
        [helper.make_tensor_value_info("loop_out", TensorProto.FLOAT, None)],
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 15)]
    ).SerializeToString()


# Seed models for fuzz_compose. Each seed is a model pair packed for the
# harness's raw path: a 4-byte big-endian length prefix for m1's serialized
# bytes, followed by m2's serialized bytes, then a trailing toggle byte.
# Toggle 0x00 = raw path + derived io_map; 0x01 also sets prefix1/prefix2.
# The harness derives the io_map from m1's output names and m2's input names,
# so each pair reaches merge_models logic on the first iteration.
def _compose_value_info(name: str):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, ["N", "M"])


def _compose_model(graph):
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 15)])


def _compose_pack(m1, m2, toggle: int) -> bytes:
    b1: bytes = m1.SerializeToString()
    b2: bytes = m2.SerializeToString()
    return struct.pack(">I", len(b1)) + b1 + b2 + bytes([toggle])


def _compose_linear() -> bytes:
    """Simplest valid merge: Relu output -> Sigmoid input (derived io_map)."""
    m1 = _compose_model(
        helper.make_graph(
            [helper.make_node("Relu", ["X"], ["Y"])],
            "m1",
            [_compose_value_info("X")],
            [_compose_value_info("Y")],
        )
    )
    m2 = _compose_model(
        helper.make_graph(
            [helper.make_node("Sigmoid", ["Yin"], ["Z"])],
            "m2",
            [_compose_value_info("Yin")],
            [_compose_value_info("Z")],
        )
    )
    return _compose_pack(m1, m2, 0x00)


def _compose_multi() -> bytes:
    """Three outputs wired to three inputs; exercises variadic io_map."""
    m1 = _compose_model(
        helper.make_graph(
            [
                helper.make_node("Add", ["X", "X"], ["O0"]),
                helper.make_node("Sub", ["X", "X"], ["O1"]),
                helper.make_node("Mul", ["X", "X"], ["O2"]),
            ],
            "m1",
            [_compose_value_info("X")],
            [_compose_value_info(n) for n in ("O0", "O1", "O2")],
        )
    )
    m2 = _compose_model(
        helper.make_graph(
            [
                helper.make_node("Add", ["I0", "I1"], ["S"]),
                helper.make_node("Mul", ["S", "I2"], ["D"]),
            ],
            "m2",
            [_compose_value_info(n) for n in ("I0", "I1", "I2")],
            [_compose_value_info("D")],
        )
    )
    return _compose_pack(m1, m2, 0x00)


def _compose_if() -> bytes:
    """m2 contains an If whose branches reference its input; exercises the
    recursive connect_io subgraph rewrite during merge.
    """
    m1 = _compose_model(
        helper.make_graph(
            [helper.make_node("Relu", ["X"], ["Y"])],
            "m1",
            [_compose_value_info("X")],
            [_compose_value_info("Y")],
        )
    )
    cond = helper.make_node(
        "Constant",
        [],
        ["c"],
        value=helper.make_tensor("cv", TensorProto.BOOL, [], [True]),
    )
    then_g = helper.make_graph(
        [helper.make_node("Relu", ["Yin"], ["tr"])],
        "then",
        [],
        [_compose_value_info("tr")],
    )
    else_g = helper.make_graph(
        [helper.make_node("Neg", ["Yin"], ["er"])],
        "else",
        [],
        [_compose_value_info("er")],
    )
    if_node = helper.make_node(
        "If", ["c"], ["Z"], then_branch=then_g, else_branch=else_g
    )
    m2 = _compose_model(
        helper.make_graph(
            [cond, if_node],
            "m2",
            [_compose_value_info("Yin")],
            [_compose_value_info("Z")],
        )
    )
    return _compose_pack(m1, m2, 0x00)


def _compose_prefix() -> bytes:
    """Identical model merged with itself; all names collide, so the seed
    sets the prefix toggle (0x01) to exercise add_prefix collision
    resolution.
    """
    graph = helper.make_graph(
        [helper.make_node("Add", ["A", "B"], ["C"])],
        "m",
        [_compose_value_info("A"), _compose_value_info("B")],
        [_compose_value_info("C")],
    )
    m1 = _compose_model(graph)
    m2 = _compose_model(graph)
    return _compose_pack(m1, m2, 0x01)


def _write_zip(path: str, entries: Mapping[str, bytes | str]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, entry in entries.items():
            zf.writestr(name, entry.encode() if isinstance(entry, str) else entry)


_USAGE = "Usage: {prog} <version_converter_out.zip> <parser_out.zip> <checker_out.zip> <shape_inference_out.zip> [compose_out.zip]\n"


def main() -> int:
    if len(sys.argv) not in (5, 6):
        sys.stderr.write(_USAGE.format(prog=sys.argv[0]))
        return 2
    version_converter_out = sys.argv[1]
    parser_out = sys.argv[2]
    checker_out = sys.argv[3]
    shape_inference_out = sys.argv[4]
    compose_out = sys.argv[5] if len(sys.argv) == 6 else None

    version_converter_seeds = {
        "cast_9_missing_input.onnx": _make_model(
            "Cast", 9, [], {"to": TensorProto.FLOAT}
        ),
        "softmax_12_missing_input.onnx": _make_model("Softmax", 12, []),
        "softmax_13_missing_input.onnx": _make_model("Softmax", 13, []),
        "upsample_6_missing_input.onnx": _make_model("Upsample", 6, []),
        "upsample_9_missing_scales.onnx": _make_model("Upsample", 9, ["X"]),
        "upsample_9_valid.onnx": _make_model("Upsample", 9, ["X", "scales"]),
    }

    # Seed models for fuzz_checker: valid serialized ModelProtos covering a
    # range of op types and opset versions so the checker reaches real
    # validation logic rather than dying at protobuf parse on every iteration.
    checker_seeds = {
        "relu_15.onnx": _make_model("Relu", 15, ["X"]),
        "sigmoid_13.onnx": _make_model("Sigmoid", 13, ["X"]),
        "tanh_13.onnx": _make_model("Tanh", 13, ["X"]),
        "abs_13.onnx": _make_model("Abs", 13, ["X"]),
        "cast_19.onnx": _make_model("Cast", 19, ["X"], {"to": TensorProto.INT64}),
        "softmax_13.onnx": _make_model("Softmax", 13, ["X"]),
    }

    # Seed models for fuzz_shape_inference covering both the per-op dispatch
    # table (Concat / MatMul / Reshape data propagation, unary chains) and the
    # recursive subgraph visitor (If / Loop). Each seed is a complete
    # serialized ModelProto with no trailing bytes, so the harness's raw path
    # loads it directly via onnx.load_model_from_string.
    shape_inference_seeds = {
        "linear_relu_sigmoid.onnx": _si_linear(),
        "concat_axis0.onnx": _si_concat(),
        "matmul_4x8_8x2.onnx": _si_matmul(),
        "reshape_2x4_to_8x1.onnx": _si_reshape(),
        "if_then_else.onnx": _si_if(),
        "loop_scan_output.onnx": _si_loop(),
    }

    _write_zip(version_converter_out, version_converter_seeds)
    _write_zip(parser_out, _PARSER_SEEDS)
    _write_zip(checker_out, checker_seeds)
    _write_zip(shape_inference_out, shape_inference_seeds)

    if compose_out is not None:
        # Seed models for fuzz_compose covering: a minimal valid merge, a
        # variadic three-pair io_map, an If subgraph (recursive connect_io
        # rewrite), and a full name collision resolved via prefixing.
        compose_seeds = {
            "compose_linear.bin": _compose_linear(),
            "compose_multi.bin": _compose_multi(),
            "compose_if.bin": _compose_if(),
            "compose_prefix.bin": _compose_prefix(),
        }
        _write_zip(compose_out, compose_seeds)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
