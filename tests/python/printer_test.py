# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
import pytest

import onnx
from onnx import helper, numpy_helper, parser, printer


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

    def test_float16_initializer_roundtrip(self) -> None:
        values = np.array([1.0, -2.0, 0.5], dtype=np.float16)
        initializer = numpy_helper.from_array(values, name="weights")
        graph = helper.make_graph([], "graph", [], [], [initializer])

        text = printer.to_text(graph)

        assert "float16[3] weights = {15360,49152,14336}" in text
        parsed = parser.parse_graph(text)
        np.testing.assert_array_equal(
            numpy_helper.to_array(parsed.initializer[0]), values
        )

    def test_to_text_unsupported_type_raises(self) -> None:
        # to_text dispatches on proto type and raises TypeError for unsupported
        # arguments. Use a proto type outside the supported set.
        with pytest.raises(TypeError):
            printer.to_text(onnx.AttributeProto())
