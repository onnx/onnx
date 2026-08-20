# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
import pytest

import onnx
from onnx import helper, numpy_helper, parser, printer


def print_initializer(initializer: onnx.TensorProto) -> str:
    """Print a graph holding `initializer` and nothing else."""
    return printer.to_text(helper.make_graph([], "graph", [], [], [initializer]))


class TestBasicFunctions:
    def check_graph(self, graph: onnx.GraphProto) -> None:
        assert len(graph.node) == 3
        assert graph.node[0].op_type == "MatMul"
        assert graph.node[1].op_type == "Add"
        assert graph.node[2].op_type == "Softmax"

    def test_parse_graph(self) -> None:
        text0 = """
           agraph (float[N, 128] X, float[128,10] W, float[10] B) => (float[N] C)
           {
              T = MatMul(X, W)
              S = Add(T, B)
              C = Softmax(S)
           }
           """
        graph1 = parser.parse_graph(text0)
        text1 = printer.to_text(graph1)
        graph2 = parser.parse_graph(text1)
        text2 = printer.to_text(graph2)
        # Note that text0 and text1 should be semantically-equivalent, but may differ
        # in white-space and other syntactic sugar. However, we expect text1 and text2
        # to be identical.
        assert text1 == text2
        self.check_graph(graph2)

    def test_quoted_string_symbolic_dim_roundtrip(self) -> None:
        # Test that non-identifier dim_params are printed as quoted strings and
        # can be parsed back (round-trip).
        text0 = 'agraph (float["M + N"] x) => (float["M + N"] y) { y = Identity(x) }'
        graph1 = parser.parse_graph(text0)
        text1 = printer.to_text(graph1)
        graph2 = parser.parse_graph(text1)
        text2 = printer.to_text(graph2)
        assert text1 == text2
        # Verify that "M + N" is preserved as a quoted string in the printed output
        assert '"M + N"' in text1

    @pytest.mark.parametrize(
        "type_text",
        [
            "opaque(test.domain,MyType)",
            "opaque(MyType)",
            "opaque()",
        ],
    )
    def test_opaque_type_roundtrip(self, type_text: str) -> None:
        # Test that Opaque types (added, along with this test, to illustrate
        # producing/consuming custom types not defined by the ONNX spec) can
        # be parsed and printed, and survive a parse/print round-trip.
        text0 = f"agraph ({type_text} x) => ({type_text} y) {{ y = Identity(x) }}"
        graph1 = parser.parse_graph(text0)
        assert graph1.input[0].type.WhichOneof("value") == "opaque_type"
        text1 = printer.to_text(graph1)
        graph2 = parser.parse_graph(text1)
        text2 = printer.to_text(graph2)
        assert text1 == text2
        assert graph2.input[0].type == graph1.input[0].type

    def test_parse_node_roundtrip(self) -> None:
        # Regression test for #7944: parse_node accepts NodeProto text but
        # printer.to_text(NodeProto) raised TypeError because NodeProto was
        # not handled in the dispatch.
        text0 = "C = Softmax(S)"
        node1 = parser.parse_node(text0)
        text1 = printer.to_text(node1)
        node2 = parser.parse_node(text1)
        text2 = printer.to_text(node2)
        assert text1 == text2
        assert node2.op_type == "Softmax"
        assert list(node2.output) == ["C"]
        assert list(node2.input) == ["S"]

    @pytest.mark.parametrize(
        ("dtype", "values", "expected"),
        [
            (np.float16, [1.0, -2.0, 0.5], "{15360,49152,14336}"),
            (np.int8, [1, -2, 127, -128], "{1,-2,127,-128}"),
            (np.uint8, [0, 1, 255], "{0,1,255}"),
            (np.bool_, [True, False, True], "{1,0,1}"),
            (np.int16, [1, -2, 32767, -32768], "{1,-2,32767,-32768}"),
            (np.uint16, [1, 2, 65535], "{1,2,65535}"),
            (np.int32, [1, -2, 2147483647], "{1,-2,2147483647}"),
            (np.uint32, [1, 2, 4294967295], "{1,2,4294967295}"),
            (np.int64, [1, -2, 9223372036854775807], "{1,-2,9223372036854775807}"),
            (np.uint64, [1, 2, 18446744073709551615], "{1,2,18446744073709551615}"),
            (np.float32, [1.5, -2.5, 0.0], "{1.5,-2.5,0}"),
            (np.float64, [1.5, -2.5, 0.0], "{1.5,-2.5,0}"),
        ],
    )
    def test_raw_initializer_roundtrip(self, dtype, values, expected) -> None:
        array = np.array(values, dtype=dtype)

        text = print_initializer(numpy_helper.from_array(array, name="weights"))

        assert expected in text
        parsed = parser.parse_graph(text)
        np.testing.assert_array_equal(
            numpy_helper.to_array(parsed.initializer[0]), array
        )

    # Bit patterns printed into int32_data; the float16 row is the non-raw one.
    @pytest.mark.parametrize(
        ("data_type", "data", "raw", "expected"),
        [
            (
                onnx.TensorProto.BFLOAT16,
                b"\x80\x3f\x00\xc0\x00\x3f",
                True,
                [16256, 49152, 16128],
            ),
            (onnx.TensorProto.FLOAT16, [1.0, -2.0, 0.5], False, [15360, 49152, 14336]),
            *[
                (dtype, b"\x38\xc0\x30", True, [56, 192, 48])
                for dtype in (
                    onnx.TensorProto.FLOAT8E4M3FN,
                    onnx.TensorProto.FLOAT8E4M3FNUZ,
                    onnx.TensorProto.FLOAT8E5M2,
                    onnx.TensorProto.FLOAT8E5M2FNUZ,
                    onnx.TensorProto.FLOAT8E8M0,
                )
            ],
        ],
    )
    def test_initializer_prints_int32_data(
        self, data_type, data, raw, expected
    ) -> None:
        initializer = helper.make_tensor("weights", data_type, [3], data, raw=raw)

        text = print_initializer(initializer)

        assert "{" + ",".join(map(str, expected)) + "}" in text
        assert list(parser.parse_graph(text).initializer[0].int32_data) == expected

    @pytest.mark.parametrize(
        ("data_type", "dims", "raw_data"),
        [
            (onnx.TensorProto.FLOAT16, [4], b"\x00\x3c"),  # too few bytes
            (onnx.TensorProto.FLOAT16, [2], b"\x00\x3c\x00\xc0\x00\x38"),  # too many
            (onnx.TensorProto.FLOAT16, [1], b"\x00\x3c\x00"),  # ragged
            (onnx.TensorProto.FLOAT, [2], b"\x00\x00\x80\x3f"),  # too few bytes
            (
                onnx.TensorProto.FLOAT,
                [1],
                b"\x00\x00\x80\x3f\x00\x00\x80\x3f",
            ),  # too many
            (onnx.TensorProto.FLOAT, [1], b"\x00\x00\x80\x3f\x00"),  # ragged
        ],
    )
    def test_raw_data_size_mismatch_raises(self, data_type, dims, raw_data) -> None:
        # Printing a mis-sized tensor would emit text that re-parses as valid.
        initializer = onnx.TensorProto(name="weights", data_type=data_type)
        initializer.dims.extend(dims)
        initializer.raw_data = raw_data

        with pytest.raises(
            onnx.shape_inference.InferenceError, match="Data size mismatch"
        ):
            print_initializer(initializer)

    @pytest.mark.parametrize(
        ("data", "raw"), [(b"\x21\x43", True), ([1, 2, 3, 4], False)]
    )
    def test_undecodable_type_prints_placeholder(self, data, raw) -> None:
        # Not decoded; "..." fails to re-parse rather than losing data silently.
        initializer = helper.make_tensor(
            "weights", onnx.TensorProto.INT4, [4], data, raw=raw
        )

        assert "..." in print_initializer(initializer)

    def test_to_text_unsupported_type_raises(self) -> None:
        # to_text dispatches on proto type and raises TypeError for unsupported
        # arguments. Use a proto type outside the supported set.
        with pytest.raises(TypeError):
            printer.to_text(onnx.AttributeProto())
