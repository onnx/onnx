from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from onnx import checker, helper, ModelProto, TensorProto, GraphProto, NodeProto
from typing import Sequence, Text, Tuple, List, Callable
from onnx import numpy_helper

import numpy as np  # type: ignore

import onnx.version_converter
import unittest


class TestVersionConverter(unittest.TestCase):

    def _converted(self, graph, initial_version, target_version):  # type: (GraphProto, OpSetID, OpSetID, int) -> ModelProto
        orig_model = helper.make_model(graph, producer_name='onnx-test')
        # print(type(orig_model))
        converted_model = onnx.version_converter.convert_version(orig_model,
                initial_version, target_version)
        checker.check_model(converted_model)
        return converted_model

    # fn is a function that takes a single node as argument
    def _visit_all_nodes_recursive(self, graph, fn):  # type: (GraphProto, Callable[[NodeProto], None]) -> None
        for node in graph.node:
            fn(node)
            for attr in node.attribute:
                if attr.g is not None:
                    self._visit_all_nodes_recursive(attr.g, fn)
                if len(attr.graphs):
                    for gr in attr.graphs:
                        self._visit_all_nodes_recursive(gr, fn)

    # Test 1: Backwards Incompatible Conversion: Add: 8 -> 2
    def test_backwards_incompatible(self):  # type: () -> None
        def test():
            nodes = [helper.make_node('Add', ["X", "X2"], ["Y"])]
            graph = helper.make_graph(
                nodes,
                "test",
                [helper.make_tensor_value_info("X", TensorProto.FLOAT, (5,)),
                    helper.make_tensor_value_info("X2", TensorProto.FLOAT, (5,))],
                [helper.make_tensor_value_info("Y", TensorProto.FLOAT, (5,))])
            converted_model = self._converted(graph, helper.make_operatorsetid(
                    "ai.onnx", 8), helper.make_operatorsetid("ai.onnx", 2))
        self.assertRaises(RuntimeError, test)

    # Test 2: Backwards Compatible Conversion: Add: 8 -> 7
    def test_backwards_compatible(self):  # type: () -> None
        nodes = [helper.make_node('Add', ["X", "X2"], ["Y"])]
        graph = helper.make_graph(
            nodes,
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (5,)),
                helper.make_tensor_value_info("X2", TensorProto.FLOAT, (5,))],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, (5,))])
        converted_model = self._converted(graph, helper.make_operatorsetid(
            "ai.onnx", 8), helper.make_operatorsetid("ai.onnx", 7))
        # TODO: Assert equality of graph and converted_model
        assert converted_model.opset_import[0].version == 7

if __name__ == '__main__':
    unittest.main()
