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
            nodes = [helper.make_node('Add', ["W", "Z"], ["shape"]),
                        helper.make_node('Reshape', ["X", "shape"], ["A"]),
                        helper.make_node('Add', ["A", "W"], ["Y"])]
            graph = helper.make_graph(
                nodes,
                "test",
                [helper.make_tensor_value_info("X", TensorProto.FLOAT, (5,)),
                    helper.make_tensor_value_info("W", TensorProto.FLOAT, (1,)),
                    helper.make_tensor_value_info("Z", TensorProto.FLOAT, (1,))],
                [helper.make_tensor_value_info("Y", TensorProto.FLOAT, (5,))])
            self._converted(graph, helper.make_operatorsetid("", 8), 2)
        self.assertRaises(RuntimeError, test)

    # Test 2: Backwards Compatible Conversion (No Adaptations): Add: 3 -> 2
    def test_backwards_compatible(self):  # type: () -> None
        nodes = [helper.make_node('Add', ["X1", "X2"], ["Y"])]
        graph = helper.make_graph(
            nodes,
            "test",
            [helper.make_tensor_value_info("X1", TensorProto.FLOAT, (5,)),
                helper.make_tensor_value_info("X2", TensorProto.FLOAT, (5,))],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, (5,))])
        converted_model = self._converted(graph, helper.make_operatorsetid(
            "", 3), 2)
        # Assert equality of graph and converted_model
        assert converted_model.graph.node[0].op_type == "Add"
        assert converted_model.opset_import[0].version == 2

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

    # Test Add Adapter: 8 -> 5
    def test_add_8_5(self):  # type: () -> None
        nodes = [helper.make_node('Add', ["X1", "X2"], ["Y"])]
        graph = helper.make_graph(
            nodes,
            "test",
            [helper.make_tensor_value_info("X1", TensorProto.FLOAT, (5,)),
                helper.make_tensor_value_info("X2", TensorProto.FLOAT, (1,))],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, (5,))])
        converted_model = self._converted(graph, helper.make_operatorsetid(
            "", 8), 5)
        # Assert equality of graph and converted_model
        assert converted_model.graph.node[0].op_type == "Add"
        assert converted_model.opset_import[0].version == 5

    # Test Add Adapter: 5 -> 8
    def test_add_5_8(self):  # type: () -> None
        nodes = [helper.make_node('Add', ["X1", "X2"], ["Y"])]
        graph = helper.make_graph(
            nodes,
            "test",
            [helper.make_tensor_value_info("X1", TensorProto.FLOAT, (5,)),
                helper.make_tensor_value_info("X2", TensorProto.FLOAT, (1,))],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, (5,))])
        converted_model = self._converted(graph, helper.make_operatorsetid(
            "", 5), 8)
        # Assert equality of graph and converted_model
        assert converted_model.graph.node[0].op_type == "Add"
        assert converted_model.opset_import[0].version == 8

    # Test Add Adapter: 5 -> 8, requiring insertion of an Unsqueeze node
    def test_add_5_8_with_unsqueeze(self):  # type: () -> None
        nodes = [helper.make_node('Add', ["X1", "X2"], ["Y"], axis=0, broadcast=1)]
        graph = helper.make_graph(
            nodes,
            "test",
            [helper.make_tensor_value_info("X1", TensorProto.FLOAT, (5, 2)),
                helper.make_tensor_value_info("X2", TensorProto.FLOAT, (5,))],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, (5,))])
        converted_model = self._converted(graph, helper.make_operatorsetid(
            "", 5), 8)
        # Assert equality of graph and converted_model
        assert converted_model.graph.node[0].op_type == "Unsqueeze"
        assert converted_model.graph.node[1].op_type == "Add"
        assert converted_model.opset_import[0].version == 8

    # Test Mul Adapter: 8 -> 5
    def test_mul_8_5(self):  # type: () -> None
        nodes = [helper.make_node('Mul', ["X1", "X2"], ["Y"])]
        graph = helper.make_graph(
            nodes,
            "test",
            [helper.make_tensor_value_info("X1", TensorProto.FLOAT, (5,)),
                helper.make_tensor_value_info("X2", TensorProto.FLOAT, (1,))],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, (5,))])
        converted_model = self._converted(graph, helper.make_operatorsetid(
            "", 8), 5)
        # Assert equality of graph and converted_model
        assert converted_model.graph.node[0].op_type == "Mul"
        assert converted_model.opset_import[0].version == 5

    # Test Mul Adapter: 5 -> 8
    def test_mul_5_8(self):  # type: () -> None
        nodes = [helper.make_node('Mul', ["X1", "X2"], ["Y"])]
        graph = helper.make_graph(
            nodes,
            "test",
            [helper.make_tensor_value_info("X1", TensorProto.FLOAT, (5,)),
                helper.make_tensor_value_info("X2", TensorProto.FLOAT, (1,))],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, (5,))])
        converted_model = self._converted(graph, helper.make_operatorsetid(
            "", 5), 8)
        # Assert equality of graph and converted_model
        assert converted_model.graph.node[0].op_type == "Mul"
        assert converted_model.opset_import[0].version == 8

    # Test Gemm Adapter: 1 -> 8
    def test_gemm_up(self):  # type: () -> None
        nodes = [helper.make_node('Gemm', ["A", "B", "C"], ["Y"])]
        graph = helper.make_graph(
            nodes,
            "test",
            [helper.make_tensor_value_info("A", TensorProto.FLOAT, (5, 5,)),
                helper.make_tensor_value_info("B", TensorProto.FLOAT, (5, 5,)),
                helper.make_tensor_value_info("C", TensorProto.FLOAT, (5, 5,))],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, (5, 5,))])
        converted_model = self._converted(graph, helper.make_operatorsetid(
            "", 1), 8)
        # Assert equality of graph and converted_model
        assert converted_model.graph.node[0].op_type == "Gemm"
        assert converted_model.opset_import[0].version == 8

    # Test Gemm Adapter: 8 -> 1
    def test_gemm_down(self):  # type: () -> None
        nodes = [helper.make_node('Gemm', ["A", "B", "C"], ["Y"])]
        graph = helper.make_graph(
            nodes,
            "test",
            [helper.make_tensor_value_info("A", TensorProto.FLOAT, (5, 5,)),
                helper.make_tensor_value_info("B", TensorProto.FLOAT, (5, 5,)),
                helper.make_tensor_value_info("C", TensorProto.FLOAT, (5, 5,))],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, (5, 5,))])
        converted_model = self._converted(graph, helper.make_operatorsetid(
            "", 8), 1)
        # Assert equality of graph and converted_model
        assert converted_model.graph.node[0].op_type == "Gemm"
        assert converted_model.opset_import[0].version == 1

    # Test Relu Adapter: 5 -> 7
    def test_relu_5_7(self):  # type: () -> None
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
    def test_relu_7_5(self):  # type: () -> None
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

    # Test BatchNormalization Adapter: 8 -> 5
    def test_batch_normalization_8_5(self):  # type: () -> None
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
            "", 8), 5)
        # Assert equality of graph and converted_model
        assert converted_model.graph.node[0].op_type == "BatchNormalization"
        assert converted_model.opset_import[0].version == 5

    # Test BatchNormalization Adapter: 5 -> 8
    def test_batch_normalization_5_8(self):  # type: () -> None
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
            "", 5), 8)
        # Assert equality of graph and converted_model
        assert converted_model.graph.node[0].op_type == "BatchNormalization"
        assert converted_model.opset_import[0].version == 8

    # Test Concat Adapter: 3 -> 5
    def test_concat_3_5(self):  # type: () -> None
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
    def test_concat_5_3(self):  # type: () -> None
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
    def test_reshape_6_4(self):  # type: () -> None
        nodes = [helper.make_node('Constant', [], ["shape"],
                    value=helper.make_tensor("", TensorProto.INT64, [1],
                        [5])),
                    helper.make_node('Reshape', ["X", "shape"], ["Y"])]
        graph = helper.make_graph(
            nodes,
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (5,))],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, (5,))])
        converted_model = self._converted(graph, helper.make_operatorsetid(
            "", 6), 4)
        # Assert equality of graph and converted_model
        assert converted_model.graph.node[0].op_type == "Reshape"
        assert converted_model.opset_import[0].version == 4

    # Test Reshape Adapter: 4 -> 6
    def test_reshape_4_6(self):  # type: () -> None
        nodes = [helper.make_node('Reshape', ["X"], ["Y"], shape=[5])]
        graph = helper.make_graph(
            nodes,
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (5,))],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, (5,))])
        converted_model = self._converted(graph, helper.make_operatorsetid(
            "", 4), 6)
        # Assert equality of graph and converted_model
        assert converted_model.graph.node[0].op_type == "Reshape"
        assert converted_model.opset_import[0].version == 6

    # Test Sum Adapter: 5 -> 8
    def test_sum_5_8(self):  # type: () -> None
        nodes = [helper.make_node('Sum', ["data_0", "data_1", "data_2",
            "data_3", "data_4"], ["sum"])]
        graph = helper.make_graph(
            nodes,
            "test",
            [helper.make_tensor_value_info("data_0", TensorProto.FLOAT, (5,)),
                helper.make_tensor_value_info("data_1", TensorProto.FLOAT, (5,)),
                helper.make_tensor_value_info("data_2", TensorProto.FLOAT, (5,)),
                helper.make_tensor_value_info("data_3", TensorProto.FLOAT, (5,)),
                helper.make_tensor_value_info("data_4", TensorProto.FLOAT, (5,))],
            [helper.make_tensor_value_info("sum", TensorProto.FLOAT, (5,))])
        converted_model = self._converted(graph, helper.make_operatorsetid(
            "", 5), 7)
        # Assert equality of graph and converted_model
        assert converted_model.graph.node[0].op_type == "Sum"
        assert converted_model.opset_import[0].version == 7

    # Test Sum Adapter: 8 -> 5
    def test_sum_8_5(self):  # type: () -> None
        nodes = [helper.make_node('Sum', ["data_0", "data_1", "data_2",
            "data_3", "data_4"], ["sum"])]
        graph = helper.make_graph(
            nodes,
            "test",
            [helper.make_tensor_value_info("data_0", TensorProto.FLOAT, (5,)),
                helper.make_tensor_value_info("data_1", TensorProto.FLOAT, (5,)),
                helper.make_tensor_value_info("data_2", TensorProto.FLOAT, (5,)),
                helper.make_tensor_value_info("data_3", TensorProto.FLOAT, (5,)),
                helper.make_tensor_value_info("data_4", TensorProto.FLOAT, (5,))],
            [helper.make_tensor_value_info("sum", TensorProto.FLOAT, (5,))])
        converted_model = self._converted(graph, helper.make_operatorsetid(
            "", 8), 5)
        # Assert equality of graph and converted_model
        assert converted_model.graph.node[0].op_type == "Sum"
        assert converted_model.opset_import[0].version == 5

    # Test AveragePool Adapter: 1 -> 8
    def test_averagepool_up(self):  # type: () -> None
        nodes = [helper.make_node('AveragePool', ["X"], ["Y"], kernel_shape=[1, 1])]
        graph = helper.make_graph(
            nodes,
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (5, 5,))],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, (5, 5,))])
        converted_model = self._converted(graph, helper.make_operatorsetid(
            "", 1), 8)
        # Assert equality of graph and converted_model
        assert converted_model.graph.node[0].op_type == "AveragePool"
        assert converted_model.opset_import[0].version == 8

    # Test AveragePool Adapter: 8 -> 1
    def test_averagepool_down(self):  # type: () -> None
        nodes = [helper.make_node('AveragePool', ["X"], ["Y"], kernel_shape=[1, 1])]
        graph = helper.make_graph(
            nodes,
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (5, 5,))],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, (5, 5,))])
        converted_model = self._converted(graph, helper.make_operatorsetid(
            "", 8), 1)
        # Assert equality of graph and converted_model
        assert converted_model.graph.node[0].op_type == "AveragePool"
        assert converted_model.opset_import[0].version == 1

    # Test Dropout Adapter: 1 -> 8
    def test_dropout_up(self):  # type: () -> None
        nodes = [helper.make_node('Dropout', ["data"], ["output"], is_test=1)]
        graph = helper.make_graph(
            nodes,
            "test",
            [helper.make_tensor_value_info("data", TensorProto.FLOAT, (5, 5,))],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, (5, 5,))])
        converted_model = self._converted(graph, helper.make_operatorsetid(
            "", 1), 8)
        # Assert equality of graph and converted_model
        assert converted_model.graph.node[0].op_type == "Dropout"
        assert converted_model.opset_import[0].version == 8

    # Test Dropout Adapter: 8 -> 1
    def test_dropout_down(self):  # type: () -> None
        nodes = [helper.make_node('Dropout', ["data"], ["output"])]
        graph = helper.make_graph(
            nodes,
            "test",
            [helper.make_tensor_value_info("data", TensorProto.FLOAT, (5, 5,))],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, (5, 5,))])
        converted_model = self._converted(graph, helper.make_operatorsetid(
            "", 8), 1)
        # Assert equality of graph and converted_model
        assert converted_model.graph.node[0].op_type == "Dropout"
        assert converted_model.opset_import[0].version == 1

    # Test MaxPool Adapter: 1 -> 8
    def test_maxpool_up(self):  # type: () -> None
        nodes = [helper.make_node('MaxPool', ["X"], ["Y"], kernel_shape=[1, 1])]
        graph = helper.make_graph(
            nodes,
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (5, 5,))],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, (5, 5,))])
        converted_model = self._converted(graph, helper.make_operatorsetid(
            "", 1), 8)
        # Assert equality of graph and converted_model
        assert converted_model.graph.node[0].op_type == "MaxPool"
        assert converted_model.opset_import[0].version == 8

    # Test MaxPool Adapter: 8 -> 1
    def test_maxpool_down(self):  # type: () -> None
        nodes = [helper.make_node('MaxPool', ["X"], ["Y"], kernel_shape=[1, 1])]
        graph = helper.make_graph(
            nodes,
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (5, 5,))],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, (5, 5,))])
        converted_model = self._converted(graph, helper.make_operatorsetid(
            "", 8), 1)
        # Assert equality of graph and converted_model
        assert converted_model.graph.node[0].op_type == "MaxPool"
        assert converted_model.opset_import[0].version == 1


if __name__ == '__main__':
    unittest.main()
