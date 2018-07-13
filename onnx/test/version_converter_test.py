from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from onnx import checker, helper, ModelProto, TensorProto, GraphProto, NodeProto, OperatorSetIdProto
from typing import Sequence, Text, Tuple, List, Callable
from onnx import numpy_helper

import numpy as np  # type: ignore

import onnx.version_converter
import unittest


class TestVersionConverter(unittest.TestCase):

    def _converted(
            self,
            graph,  # type: GraphProto
            initial_version,  # type: OperatorSetIdProto
            target_version  # type: int
    ):  # type: (...) -> ModelProto
        orig_model = helper.make_model(graph, producer_name='onnx-test', opset_imports=[initial_version])
        # print(type(orig_model))
        converted_model = onnx.version_converter.convert_version(orig_model,
                target_version)
        checker.check_model(converted_model)
        return converted_model

    # Test 1: Backwards Incompatible Conversion: Reshape: 8 -> 2
    def test_backwards_incompatible(self):  # type: () -> None
        def test():  # type: () -> None
            nodes = [helper.make_node('Reshape', ["X", "shape"], ["Y"])]
            graph = helper.make_graph(
                nodes,
                "test",
                [helper.make_tensor_value_info("X", TensorProto.FLOAT, (5,)),
                    helper.make_tensor_value_info("shape", TensorProto.FLOAT, (1,))],
                [helper.make_tensor_value_info("Y", TensorProto.FLOAT, (5,))])
            self._converted(graph, helper.make_operatorsetid("", 8), 2)
        self.assertRaises(RuntimeError, test)

    # Test 2: Backwards Compatible Conversion: Add: 8 -> 7
    def test_backwards_compatible(self):  # type: () -> None
        nodes = [helper.make_node('Add', ["X1", "X2"], ["Y"])]
        graph = helper.make_graph(
            nodes,
            "test",
            [helper.make_tensor_value_info("X1", TensorProto.FLOAT, (5,)),
                helper.make_tensor_value_info("X2", TensorProto.FLOAT, (5,))],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, (5,))])
        converted_model = self._converted(graph, helper.make_operatorsetid(
            "", 8), 7)
        # Assert equality of graph and converted_model
        assert converted_model.graph.node[0].op_type == "Add"
        assert converted_model.opset_import[0].version == 7

    # Test 3: Non-Existent Op Conversion: Cos: 8 -> 6
    def test_non_existent_op(self):  # type: () -> None
        def test():  # type: () -> None
            nodes = [helper.make_node('Cos', ["X"], ["Y"])]
            graph = helper.make_graph(
                nodes,
                "test",
                [helper.make_tensor_value_info("X", TensorProto.FLOAT, (5,))],
                [helper.make_tensor_value_info("Y", TensorProto.FLOAT, (5,))])
            self._converted(graph, helper.make_operatorsetid("", 8), 6)
        self.assertRaises(RuntimeError, test)


if __name__ == '__main__':
    unittest.main()
