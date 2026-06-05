# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import unittest

import onnx
from onnx import parser, printer


class TestBasicFunctions(unittest.TestCase):
    def check_graph(self, graph: onnx.GraphProto) -> None:
        self.assertEqual(len(graph.node), 3)
        self.assertEqual(graph.node[0].op_type, "MatMul")
        self.assertEqual(graph.node[1].op_type, "Add")
        self.assertEqual(graph.node[2].op_type, "Softmax")

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
        self.assertEqual(text1, text2)
        self.check_graph(graph2)

    def test_quoted_string_symbolic_dim_roundtrip(self) -> None:
        # Test that non-identifier dim_params are printed as quoted strings and
        # can be parsed back (round-trip).
        text0 = 'agraph (float["M + N"] x) => (float["M + N"] y) { y = Identity(x) }'
        graph1 = parser.parse_graph(text0)
        text1 = printer.to_text(graph1)
        graph2 = parser.parse_graph(text1)
        text2 = printer.to_text(graph2)
        self.assertEqual(text1, text2)
        # Verify that "M + N" is preserved as a quoted string in the printed output
        self.assertIn('"M + N"', text1)

    def test_parse_node_roundtrip(self) -> None:
        # Regression test for #7944: parse_node accepts NodeProto text but
        # printer.to_text(NodeProto) raised TypeError because NodeProto was
        # not handled in the dispatch.
        text0 = "C = Softmax(S)"
        node1 = parser.parse_node(text0)
        text1 = printer.to_text(node1)
        node2 = parser.parse_node(text1)
        text2 = printer.to_text(node2)
        self.assertEqual(text1, text2)
        self.assertEqual(node2.op_type, "Softmax")
        self.assertEqual(list(node2.output), ["C"])
        self.assertEqual(list(node2.input), ["S"])

    def test_to_text_unsupported_type_raises(self) -> None:
        # to_text dispatches on proto type and raises TypeError for unsupported
        # arguments. Use a proto type outside the supported set.
        with self.assertRaises(TypeError):
            printer.to_text(onnx.AttributeProto())


class TestRoundTrip(unittest.TestCase):
    """parse -> print -> parse -> print round-trip checks.

    These moved here from the C++ parser_test.cc ``Parse`` helper when the text
    printer was reimplemented in pure Python (onnx/printer.py). Like the C++
    check, we do not compare the printed text to the original input (they differ
    in white-space and syntactic sugar); instead we print once, parse that, print
    again, and require the two printed forms to be identical.
    """

    def _check(self, parse_fn, text0: str):
        proto1 = parse_fn(text0)
        text1 = printer.to_text(proto1)
        proto2 = parse_fn(text1)
        text2 = printer.to_text(proto2)
        self.assertEqual(text1, text2)
        return proto2

    def test_model_roundtrip(self) -> None:
        self._check(
            parser.parse_model,
            """
            <
              ir_version: 7,
              opset_import: [ "ai.onnx.ml" : 10 ],
              producer_name: "ParserTest",
              producer_version: "1.0",
              domain: "ai.onnx.ml",
              model_version: 1,
              doc_string: "A parser test case model.",
              metadata_props: [ "somekey" : "somevalue", "key2" : "value2" ]
            >
            agraph (float[N] y, float[N] z) => (float[N] w)
            {
                x = foo(y, z)
                w = bar(x, y)
            }
            """,
        )

    def test_if_subgraph_roundtrip(self) -> None:
        self._check(
            parser.parse_model,
            """
            <ir_version: 7, opset_import: [ "" : 13 ]>
            iftest (bool b, float[128] X, float[128] Y) => (float[128] Z)
            {
              Z = If (b) <
                  then_branch = g1 () => (float[128] z_then) { z_then = Identity(X) },
                  else_branch = g2 () => (float[128] z_else) { z_else = Identity(Y) }
                  >
            }
            """,
        )

    def test_functions_roundtrip(self) -> None:
        self._check(
            parser.parse_model,
            """
            <ir_version: 8, opset_import: [ "" : 10, "local" : 1 ]>
            agraph (float[N, 128] X, float[128,10] W, float[10] B) => (float[N] C)
            {
              T = local.foo (X, W, B)
              C = local.square(T)
            }
            <opset_import: [ "" : 10 ], domain: "local", doc_string: "Function foo.">
            foo (x, w, b) => (c) {
              T = MatMul(x, w)
              S = Add(T, b)
              c = Softmax(S)
            }
            <opset_import: [ "" : 10 ], domain: "local">
            square (x) => (y) { y = Mul (x, x) }
            """,
        )

    def test_function_with_attr_ref_roundtrip(self) -> None:
        # Exercises the attr-ref printer path (name: type = @ref) and a declared
        # function attribute with a default value.
        self._check(
            parser.parse_function,
            """
            <domain: "custom_domain", opset_import: [ "" : 15]>
            foo <alpha: float=4.0, gamma> (X) => (C)
            {
                ca = Constant<value_float: float=@alpha>()
                cg = Constant<value_float: float=@gamma>()
                cax = Mul(ca, X)
                C = Add(cax, cg)
            }
            """,
        )

    def test_graph_initializers_roundtrip(self) -> None:
        graph = self._check(
            parser.parse_graph,
            """
            agraph (float[N] x) => (float[N] y)
            <int32[2] i = {3, 4}, float[2] f = {1.5, 2.5}, float[3] s>
            {
              y = Identity(x)
            }
            """,
        )
        # i and f carry values (initializers); s has no value (value_info).
        self.assertEqual(len(graph.initializer), 2)
        self.assertEqual(len(graph.value_info), 1)

    def test_node_attributes_roundtrip(self) -> None:
        node = self._check(
            parser.parse_node,
            'r = foo <d = [5, 10], e = [0.55, 0.66], f = ["str1", "str2"], '
            'g = "txt", h = 3, t = float[2] {1.0, 2.0}> (y, z)',
        )
        self.assertEqual(node.op_type, "foo")

    def test_domain_qualified_node_roundtrip(self) -> None:
        node = self._check(parser.parse_node, "r = com.example.foo(x, y)")
        self.assertEqual(node.domain, "com.example")

    def test_types_model_roundtrip(self) -> None:
        # seq / optional / sparse_tensor / map value-info types.
        self._check(
            parser.parse_model,
            """
            <ir_version: 8, opset_import: [ "" : 17 ]>
            g (float[N] x) => (float[N] y)
            <seq(float[]) sq, optional(float[2]) op, sparse_tensor(float[8]) sp>
            {
              y = Identity(x)
            }
            """,
        )

    def test_quoted_identifiers_roundtrip(self) -> None:
        graph = self._check(
            parser.parse_graph,
            'agraph (float["M + N"] "in put") => (float["M + N"] "out put") '
            '{ "out put" = Identity("in put") }',
        )
        text = printer.to_text(graph)
        self.assertIn('"M + N"', text)
        self.assertIn('"in put"', text)


if __name__ == "__main__":
    unittest.main()
