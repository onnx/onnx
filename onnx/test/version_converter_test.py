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
    # TODO: Rewrite test to provide dynamic shape parameter, preventing conversion
    # def test_backwards_incompatible(self):  # type: () -> None
    #     def test():  # type: () -> None
    #         nodes = [helper.make_node('Reshape', ["X", "shape"], ["Y"])]
    #         graph = helper.make_graph(
    #             nodes,
    #             "test",
    #             [helper.make_tensor_value_info("X", TensorProto.FLOAT, (5,)),
    #                 helper.make_tensor_value_info("shape", TensorProto.FLOAT, (1,))],
    #             [helper.make_tensor_value_info("Y", TensorProto.FLOAT, (5,))])
    #         self._converted(graph, helper.make_operatorsetid("", 8), 2)
    #     self.assertRaises(RuntimeError, test)

    # Test 2: Backwards Compatible Conversion (No Adaptations): Add: 8 -> 7
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

    # Test Add Adapter: 8 -> 6
    def test_add_7_6(self):  # type: () -> None
        nodes = [helper.make_node('Add', ["X1", "X2"], ["Y"])]
        graph = helper.make_graph(
            nodes,
            "test",
            [helper.make_tensor_value_info("X1", TensorProto.FLOAT, (5,)),
                helper.make_tensor_value_info("X2", TensorProto.FLOAT, (1,))],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, (5,))])
        converted_model = self._converted(graph, helper.make_operatorsetid(
            "", 8), 6)
        # Assert equality of graph and converted_model
        assert converted_model.graph.node[0].op_type == "Add"
        assert converted_model.opset_import[0].version == 6

    # Test Add Adapter: 6 -> 8
    def test_add_6_7(self):  # type: () -> None
        nodes = [helper.make_node('Add', ["X1", "X2"], ["Y"])]
        graph = helper.make_graph(
            nodes,
            "test",
            [helper.make_tensor_value_info("X1", TensorProto.FLOAT, (5,)),
                helper.make_tensor_value_info("X2", TensorProto.FLOAT, (1,))],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, (5,))])
        converted_model = self._converted(graph, helper.make_operatorsetid(
            "", 6), 8)
        # Assert equality of graph and converted_model
        assert converted_model.graph.node[0].op_type == "Add"
        assert converted_model.opset_import[0].version == 8

    # Test Add Adapter: 7 -> 5
    def test_add_6_5(self):  # type: () -> None
        nodes = [helper.make_node('Add', ["X1", "X2"], ["Y"])]
        graph = helper.make_graph(
            nodes,
            "test",
            [helper.make_tensor_value_info("X1", TensorProto.FLOAT, (5,)),
                helper.make_tensor_value_info("X2", TensorProto.FLOAT, (1,))],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, (5,))])
        converted_model = self._converted(graph, helper.make_operatorsetid(
            "", 7), 5)
        # Assert equality of graph and converted_model
        assert converted_model.graph.node[0].op_type == "Add"
        assert converted_model.opset_import[0].version == 5

    # Test Add Adapter: 5 -> 7
    def test_add_5_6(self):  # type: () -> None
        nodes = [helper.make_node('Add', ["X1", "X2"], ["Y"])]
        graph = helper.make_graph(
            nodes,
            "test",
            [helper.make_tensor_value_info("X1", TensorProto.FLOAT, (5,)),
                helper.make_tensor_value_info("X2", TensorProto.FLOAT, (1,))],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, (5,))])
        converted_model = self._converted(graph, helper.make_operatorsetid(
            "", 5), 7)
        # Assert equality of graph and converted_model
        assert converted_model.graph.node[0].op_type == "Add"
        assert converted_model.opset_import[0].version == 7

    # Test Mul Adapter: 8 -> 6
    def test_mul_7_6(self):  # type: () -> None
        nodes = [helper.make_node('Mul', ["X1", "X2"], ["Y"])]
        graph = helper.make_graph(
            nodes,
            "test",
            [helper.make_tensor_value_info("X1", TensorProto.FLOAT, (5,)),
                helper.make_tensor_value_info("X2", TensorProto.FLOAT, (1,))],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, (5,))])
        converted_model = self._converted(graph, helper.make_operatorsetid(
            "", 8), 6)
        # Assert equality of graph and converted_model
        assert converted_model.graph.node[0].op_type == "Mul"
        assert converted_model.opset_import[0].version == 6

    # Test Mul Adapter: 6 -> 8
    def test_mul_6_7(self):  # type: () -> None
        nodes = [helper.make_node('Mul', ["X1", "X2"], ["Y"])]
        graph = helper.make_graph(
            nodes,
            "test",
            [helper.make_tensor_value_info("X1", TensorProto.FLOAT, (5,)),
                helper.make_tensor_value_info("X2", TensorProto.FLOAT, (1,))],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, (5,))])
        converted_model = self._converted(graph, helper.make_operatorsetid(
            "", 6), 8)
        # Assert equality of graph and converted_model
        assert converted_model.graph.node[0].op_type == "Mul"
        assert converted_model.opset_import[0].version == 8

    # Test Mul Adapter: 7 -> 5
    def test_mul_6_5(self):  # type: () -> None
        nodes = [helper.make_node('Mul', ["X1", "X2"], ["Y"])]
        graph = helper.make_graph(
            nodes,
            "test",
            [helper.make_tensor_value_info("X1", TensorProto.FLOAT, (5,)),
                helper.make_tensor_value_info("X2", TensorProto.FLOAT, (1,))],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, (5,))])
        converted_model = self._converted(graph, helper.make_operatorsetid(
            "", 7), 5)
        # Assert equality of graph and converted_model
        assert converted_model.graph.node[0].op_type == "Mul"
        assert converted_model.opset_import[0].version == 5

    # Test Mul Adapter: 5 -> 7
    def test_mul_5_6(self):  # type: () -> None
        nodes = [helper.make_node('Mul', ["X1", "X2"], ["Y"])]
        graph = helper.make_graph(
            nodes,
            "test",
            [helper.make_tensor_value_info("X1", TensorProto.FLOAT, (5,)),
                helper.make_tensor_value_info("X2", TensorProto.FLOAT, (1,))],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, (5,))])
        converted_model = self._converted(graph, helper.make_operatorsetid(
            "", 5), 7)
        # Assert equality of graph and converted_model
        assert converted_model.graph.node[0].op_type == "Mul"
        assert converted_model.opset_import[0].version == 7

    # Test Relu Adapter: 5 -> 7
    def test_relu_5_6(self):  # type: () -> None
        nodes = [helper.make_node('Relu', ["X"], ["Y"])]
        graph = helper.make_graph(
            nodes,
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (5,))],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, (5,))])
        converted_model = self._converted(graph, helper.make_operatorsetid(
            "", 5), 7)
        # Assert equality of graph and converted_model
        assert converted_model.graph.node[0].op_type == "Relu"
        assert converted_model.opset_import[0].version == 7

    # Test Relu Adapter: 7 -> 5
    def test_relu_6_5(self):  # type: () -> None
        nodes = [helper.make_node('Relu', ["X"], ["Y"])]
        graph = helper.make_graph(
            nodes,
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (5,))],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, (5,))])
        converted_model = self._converted(graph, helper.make_operatorsetid(
            "", 7), 5)
        # Assert equality of graph and converted_model
        assert converted_model.graph.node[0].op_type == "Relu"
        assert converted_model.opset_import[0].version == 5

    # Test BatchNormalization Adapter: 8 -> 6
    def test_batch_normalization_7_6(self):  # type: () -> None
        nodes = [helper.make_node('BatchNormalization', ["X", "scale", "B",
            "mean", "var"], ["Y"])]
        graph = helper.make_graph(
            nodes,
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (5,)),
                helper.make_tensor_value_info("scale", TensorProto.FLOAT, (1,)),
                helper.make_tensor_value_info("B", TensorProto.FLOAT, (1,)),
                helper.make_tensor_value_info("mean", TensorProto.FLOAT, (1,)),
                helper.make_tensor_value_info("var", TensorProto.FLOAT, (1,))],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, (5,))])
        converted_model = self._converted(graph, helper.make_operatorsetid(
            "", 8), 6)
        # Assert equality of graph and converted_model
        assert converted_model.graph.node[0].op_type == "BatchNormalization"
        assert converted_model.opset_import[0].version == 6

    # Test BatchNormalization Adapter: 6 -> 8
    def test_batch_normalization_6_7(self):  # type: () -> None
        nodes = [helper.make_node('BatchNormalization', ["X", "scale", "B",
            "mean", "var"], ["Y"])]
        graph = helper.make_graph(
            nodes,
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (5,)),
                helper.make_tensor_value_info("scale", TensorProto.FLOAT, (1,)),
                helper.make_tensor_value_info("B", TensorProto.FLOAT, (1,)),
                helper.make_tensor_value_info("mean", TensorProto.FLOAT, (1,)),
                helper.make_tensor_value_info("var", TensorProto.FLOAT, (1,))],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, (5,))])
        converted_model = self._converted(graph, helper.make_operatorsetid(
            "", 6), 8)
        # Assert equality of graph and converted_model
        assert converted_model.graph.node[0].op_type == "BatchNormalization"
        assert converted_model.opset_import[0].version == 8

    # Test BatchNormalization Adapter: 7 -> 5
    def test_batch_normalization_6_5(self):  # type: () -> None
        nodes = [helper.make_node('BatchNormalization', ["X", "scale", "B",
            "mean", "var"], ["Y"])]
        graph = helper.make_graph(
            nodes,
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (5,)),
                helper.make_tensor_value_info("scale", TensorProto.FLOAT, (1,)),
                helper.make_tensor_value_info("B", TensorProto.FLOAT, (1,)),
                helper.make_tensor_value_info("mean", TensorProto.FLOAT, (1,)),
                helper.make_tensor_value_info("var", TensorProto.FLOAT, (1,))],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, (5,))])
        converted_model = self._converted(graph, helper.make_operatorsetid(
            "", 7), 5)
        # Assert equality of graph and converted_model
        assert converted_model.graph.node[0].op_type == "BatchNormalization"
        assert converted_model.opset_import[0].version == 5

    # Test BatchNormalization Adapter: 5 -> 7
    def test_batch_normalization_5_6(self):  # type: () -> None
        nodes = [helper.make_node('BatchNormalization', ["X", "scale", "B",
            "mean", "var"], ["Y"])]
        graph = helper.make_graph(
            nodes,
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (5,)),
                helper.make_tensor_value_info("scale", TensorProto.FLOAT, (1,)),
                helper.make_tensor_value_info("B", TensorProto.FLOAT, (1,)),
                helper.make_tensor_value_info("mean", TensorProto.FLOAT, (1,)),
                helper.make_tensor_value_info("var", TensorProto.FLOAT, (1,))],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, (5,))])
        converted_model = self._converted(graph, helper.make_operatorsetid(
            "", 5), 7)
        # Assert equality of graph and converted_model
        assert converted_model.graph.node[0].op_type == "BatchNormalization"
        assert converted_model.opset_import[0].version == 7

    # Test Concat Adapter: 3 -> 5
    def test_concat_3_4(self):  # type: () -> None
        nodes = [helper.make_node('Concat', ["X1", "X2", "X3",
            "X4", "X5"], ["Y"])]
        graph = helper.make_graph(
            nodes,
            "test",
            [helper.make_tensor_value_info("X1", TensorProto.FLOAT, (5,)),
                helper.make_tensor_value_info("X2", TensorProto.FLOAT, (1,)),
                helper.make_tensor_value_info("X3", TensorProto.FLOAT, (1,)),
                helper.make_tensor_value_info("X4", TensorProto.FLOAT, (1,)),
                helper.make_tensor_value_info("X5", TensorProto.FLOAT, (1,))],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, (5,))])
        converted_model = self._converted(graph, helper.make_operatorsetid(
            "", 3), 5)
        # Assert equality of graph and converted_model
        assert converted_model.graph.node[0].op_type == "Concat"
        assert converted_model.opset_import[0].version == 5

    # Test Concat Adapter: 5 -> 3
    def test_concat_4_3(self):  # type: () -> None
        nodes = [helper.make_node('Concat', ["X1", "X2", "X3",
            "X4", "X5"], ["Y"])]
        graph = helper.make_graph(
            nodes,
            "test",
            [helper.make_tensor_value_info("X1", TensorProto.FLOAT, (5,)),
                helper.make_tensor_value_info("X2", TensorProto.FLOAT, (1,)),
                helper.make_tensor_value_info("X3", TensorProto.FLOAT, (1,)),
                helper.make_tensor_value_info("X4", TensorProto.FLOAT, (1,)),
                helper.make_tensor_value_info("X5", TensorProto.FLOAT, (1,))],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, (5,))])
        converted_model = self._converted(graph, helper.make_operatorsetid(
            "", 5), 3)
        # Assert equality of graph and converted_model
        assert converted_model.graph.node[0].op_type == "Concat"
        assert converted_model.opset_import[0].version == 3

    # Test Reshape Adapter: 6 -> 4
    def test_reshape_5_4(self):  # type: () -> None
        nodes = [helper.make_node('Reshape', ["X", "shape"], ["Y"])]
        graph = helper.make_graph(
            nodes,
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (5,)),
                helper.make_tensor_value_info("shape", TensorProto.FLOAT, (1,))],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, (5,))])
        converted_model = self._converted(graph, helper.make_operatorsetid(
            "", 6), 4)
        # Assert equality of graph and converted_model
        assert converted_model.graph.node[0].op_type == "Reshape"
        assert converted_model.opset_import[0].version == 4

if __name__ == '__main__':
    unittest.main()
