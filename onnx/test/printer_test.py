# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import unittest

import onnx
from onnx import (
    TensorProto,
    checker,
    helper,
    parser,
    printer,
)


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


class TestPrintableGraph(unittest.TestCase):
    def test_initializer_with_matching_graph_input(self) -> None:
        add = helper.make_node("Add", ["X", "Y_Initializer"], ["Z"])
        value_info = [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1])]

        graph = helper.make_graph(
            [add],
            "test",
            [
                helper.make_tensor_value_info("X", TensorProto.FLOAT, [1]),
                helper.make_tensor_value_info("Y_Initializer", TensorProto.FLOAT, [1]),
            ],  # inputs
            [helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1])],  # outputs
            [
                helper.make_tensor("Y_Initializer", TensorProto.FLOAT, [1], [1])
            ],  # initializers
            doc_string=None,
            value_info=value_info,
        )

        graph_str = printer.to_text(graph)
        self.assertIn(
            "test (float[1] X, float[1] Y_Initializer) => (float[1] Z)",
            graph_str,
        )

    def test_initializer_no_matching_graph_input(self) -> None:
        add = helper.make_node("Add", ["X", "Y_Initializer"], ["Z"])
        value_info = [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1])]

        graph = helper.make_graph(
            [add],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1])],  # inputs
            [helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1])],  # outputs
            [
                helper.make_tensor("Y_Initializer", TensorProto.FLOAT, [1], [1])
            ],  # initializers
            doc_string=None,
            value_info=value_info,
        )

        graph_str = printer.to_text(graph)
        self.assertIn("test (float[1] X) => (float[1] Z)", graph_str)

    def test_unknown_dimensions(self) -> None:
        graph = helper.make_graph(
            [helper.make_node("Add", ["X", "Y_Initializer"], ["Z"])],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [None])],  # inputs
            [helper.make_tensor_value_info("Z", TensorProto.FLOAT, [None])],  # outputs
            [
                helper.make_tensor("Y_Initializer", TensorProto.FLOAT, [1], [1])
            ],  # initializers
            doc_string=None,
        )
        model = helper.make_model(graph)
        checker.check_model(model)

        graph_str = printer.to_text(graph)
        self.assertIn("test (float[?] X) => (float[?] Z)", graph_str)


if __name__ == "__main__":
    unittest.main()
