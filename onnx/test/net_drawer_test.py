# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import unittest

from onnx import TensorProto, helper

try:
    import pydot

    from onnx.tools.net_drawer import (
        OP_STYLE,
        GetOpNodeProducer,
        GetPydotGraph,
        get_op_node_producer,
        get_pydot_graph,
    )

    HAS_PYDOT = True
except ImportError:
    HAS_PYDOT = False


def _make_relu_graph():
    node = helper.make_node("Relu", inputs=["X"], outputs=["Y"])
    return helper.make_graph(
        [node],
        "relu_graph",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [3])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3])],
    )


def _make_chain_graph():
    nodes = [
        helper.make_node("Relu", inputs=["X"], outputs=["A"], name="relu"),
        helper.make_node("Add", inputs=["A", "B"], outputs=["Y"], name="add"),
    ]
    return helper.make_graph(
        nodes,
        "chain_graph",
        [
            helper.make_tensor_value_info("X", TensorProto.FLOAT, [3]),
            helper.make_tensor_value_info("B", TensorProto.FLOAT, [3]),
        ],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3])],
    )


@unittest.skipUnless(HAS_PYDOT, "pydot not installed")
class TestGetPydotGraph(unittest.TestCase):
    def test_returns_dot_instance(self) -> None:
        graph = GetPydotGraph(_make_relu_graph())
        self.assertIsInstance(graph, pydot.Dot)

    def test_simple_graph_nodes_and_edges(self) -> None:
        graph = GetPydotGraph(_make_relu_graph())
        # X blob + Relu op + Y blob = 3 nodes; X→Relu edge + Relu→Y edge = 2 edges
        self.assertEqual(len(graph.get_nodes()), 3)
        self.assertEqual(len(graph.get_edges()), 2)

    def test_empty_graph(self) -> None:
        empty = helper.make_graph([], "empty", [], [])
        graph = GetPydotGraph(empty)
        self.assertIsInstance(graph, pydot.Dot)
        self.assertEqual(len(graph.get_nodes()), 0)
        self.assertEqual(len(graph.get_edges()), 0)

    def test_name_none_does_not_raise(self) -> None:
        graph = GetPydotGraph(_make_relu_graph(), name=None)
        self.assertIsInstance(graph, pydot.Dot)

    def test_name_propagated(self) -> None:
        graph = GetPydotGraph(_make_relu_graph(), name="my_graph")
        self.assertIn("my_graph", graph.get_name())

    def test_chain_graph(self) -> None:
        graph = GetPydotGraph(_make_chain_graph())
        # X, A, B, Y blobs + Relu, Add ops = 6 nodes; 4 edges
        self.assertEqual(len(graph.get_nodes()), 6)
        self.assertEqual(len(graph.get_edges()), 4)

    def test_custom_node_producer(self) -> None:
        producer = GetOpNodeProducer(**OP_STYLE)
        graph = GetPydotGraph(_make_relu_graph(), node_producer=producer)
        self.assertIsInstance(graph, pydot.Dot)
        self.assertEqual(len(graph.get_nodes()), 3)

    def test_rankdir_options(self) -> None:
        for rankdir in ("LR", "RL", "TB", "BT"):
            graph = GetPydotGraph(_make_relu_graph(), rankdir=rankdir)
            self.assertIsInstance(graph, pydot.Dot)

    def test_optional_empty_inputs_are_skipped(self) -> None:
        # ONNX represents absent optional inputs as empty strings; they must not
        # become graph nodes or edges.
        node = helper.make_node("Clip", inputs=["X", "", ""], outputs=["Y"])
        graph = helper.make_graph(
            [node],
            "clip_graph",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [3])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3])],
        )
        pydot_graph = GetPydotGraph(graph)
        # X blob + Clip op + Y blob = 3 nodes; X→Clip + Clip→Y = 2 edges
        self.assertEqual(len(pydot_graph.get_nodes()), 3)
        self.assertEqual(len(pydot_graph.get_edges()), 2)

    def test_embed_docstring_warns_when_custom_node_producer_supplied(self) -> None:
        producer = GetOpNodeProducer(**OP_STYLE)
        with self.assertWarns(UserWarning):
            GetPydotGraph(_make_relu_graph(), node_producer=producer, embed_docstring=True)


@unittest.skipUnless(HAS_PYDOT, "pydot not installed")
class TestGetOpNodeProducer(unittest.TestCase):
    def test_returns_pydot_node(self) -> None:
        node_proto = helper.make_node("Add", inputs=["A", "B"], outputs=["C"], name="my_add")
        producer = GetOpNodeProducer(**OP_STYLE)
        node = producer(node_proto, 0)
        self.assertIsInstance(node, pydot.Node)

    def test_unnamed_op_label_contains_op_type_and_id(self) -> None:
        node_proto = helper.make_node("Relu", inputs=["X"], outputs=["Y"])
        producer = GetOpNodeProducer(**OP_STYLE)
        node = producer(node_proto, 7)
        label = node.get_name()
        self.assertIn("Relu", label)
        self.assertIn("op#7", label)

    def test_named_op_label_contains_name(self) -> None:
        node_proto = helper.make_node("Relu", inputs=["X"], outputs=["Y"], name="my_relu")
        producer = GetOpNodeProducer(**OP_STYLE)
        node = producer(node_proto, 0)
        self.assertIn("my_relu", node.get_name())

    def test_embed_docstring(self) -> None:
        node_proto = helper.make_node("Relu", inputs=["X"], outputs=["Y"])
        node_proto.doc_string = "Applies relu"
        producer = GetOpNodeProducer(embed_docstring=True, **OP_STYLE)
        node = producer(node_proto, 0)
        tooltip = node.get_tooltip()
        self.assertIsNotNone(tooltip)
        self.assertIn("Applies relu", tooltip)

    def test_embed_docstring_empty_doc_string_sets_no_tooltip(self) -> None:
        node_proto = helper.make_node("Relu", inputs=["X"], outputs=["Y"])
        producer = GetOpNodeProducer(embed_docstring=True, **OP_STYLE)
        node = producer(node_proto, 0)
        self.assertIsNone(node.get_tooltip())


@unittest.skipUnless(HAS_PYDOT, "pydot not installed")
class TestSnakeCaseAliases(unittest.TestCase):
    def test_get_op_node_producer_is_alias(self) -> None:
        self.assertIs(get_op_node_producer, GetOpNodeProducer)

    def test_get_pydot_graph_is_alias(self) -> None:
        self.assertIs(get_pydot_graph, GetPydotGraph)


if __name__ == "__main__":
    unittest.main()
