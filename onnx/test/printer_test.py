# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import onnx
from onnx import parser, printer


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

    def test_to_text_unsupported_type_raises(self) -> None:
        # to_text dispatches on proto type and raises TypeError for unsupported
        # arguments. Use a proto type outside the supported set.
        with pytest.raises(TypeError):
            printer.to_text(onnx.AttributeProto())
