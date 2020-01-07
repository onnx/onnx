from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from onnx import checker, helper, ModelProto, TensorProto, GraphProto, NodeProto, OperatorSetIdProto
from typing import Sequence, Text, Tuple, List, Callable
from onnx import numpy_helper

import numpy as np  # type: ignore
import struct

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

    # Test Sum Adapter: 7 -> 8
    def test_sum_7_8(self):  # type: () -> None
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
            "", 7), 8)
        # Assert equality of graph and converted_model
        assert converted_model.graph.node[0].op_type == "Sum"
        assert converted_model.opset_import[0].version == 8

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

    # Test Max Adapter: 7 -> 8
    def test_max_7_8(self):  # type: () -> None
        from_opset = 7
        to_opset = 8
        data_type = TensorProto.FLOAT
        data_shape = (2, 3, 4)

        nodes = [onnx.helper.make_node(
            "Max",
            inputs=["X"],
            outputs=["Y"]
        )]

        graph = helper.make_graph(
            nodes,
            "test_max",
            [onnx.helper.make_tensor_value_info("X", data_type, data_shape)],
            [onnx.helper.make_tensor_value_info("Y", data_type, data_shape)])

        converted_model = self._converted(graph, helper.make_operatorsetid("", from_opset), to_opset)

        assert converted_model.graph.node[0].op_type == "Max"
        assert converted_model.graph.output[0].type.tensor_type.elem_type == data_type
        assert converted_model.opset_import[0].version == to_opset

    # Test Min Adapter: 7 -> 8
    def test_min_7_8(self):  # type: () -> None
        from_opset = 7
        to_opset = 8
        data_type = TensorProto.FLOAT
        data_shape = (2, 3, 4)

        nodes = [onnx.helper.make_node(
            "Min",
            inputs=["X"],
            outputs=["Y"]
        )]

        graph = helper.make_graph(
            nodes,
            "test_min",
            [onnx.helper.make_tensor_value_info("X", data_type, data_shape)],
            [onnx.helper.make_tensor_value_info("Y", data_type, data_shape)])

        converted_model = self._converted(graph, helper.make_operatorsetid("", from_opset), to_opset)

        assert converted_model.graph.node[0].op_type == "Min"
        assert converted_model.graph.output[0].type.tensor_type.elem_type == data_type
        assert converted_model.opset_import[0].version == to_opset

    # Test Mean Adapter: 7 -> 8
    def test_mean_7_8(self):  # type: () -> None
        from_opset = 7
        to_opset = 8
        data_type = TensorProto.FLOAT
        data_shape = (3,)

        nodes = [onnx.helper.make_node(
            "Mean",
            inputs=["X"],
            outputs=["Y"]
        )]

        graph = helper.make_graph(
            nodes,
            "test_mean",
            [onnx.helper.make_tensor_value_info("X", data_type, data_shape)],
            [onnx.helper.make_tensor_value_info("Y", data_type, data_shape)])

        converted_model = self._converted(graph, helper.make_operatorsetid("", from_opset), to_opset)

        assert converted_model.graph.node[0].op_type == "Mean"
        assert converted_model.graph.output[0].type.tensor_type.elem_type == data_type
        assert converted_model.opset_import[0].version == to_opset

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

    # Test BatchNormalization Adapter: 8 -> 9
    def test_batch_normalization_8_9(self):  # type: () -> None
        from_opset = 8
        to_opset = 9
        data_type = TensorProto.FLOAT

        nodes = [helper.make_node(
            'BatchNormalization',
            inputs=["x", "s", "bias", "mean", "var"],
            outputs=["y"]
        )]

        input_shape = (1, 2, 1, 3)
        x = helper.make_tensor_value_info("x", data_type, input_shape)
        scale = helper.make_tensor_value_info("s", data_type, [input_shape[1]])
        B = helper.make_tensor_value_info("bias", data_type, [input_shape[1]])
        mean = helper.make_tensor_value_info("mean", data_type, [input_shape[1]])
        var = helper.make_tensor_value_info("var", data_type, [input_shape[1]])
        y = helper.make_tensor_value_info("y", data_type, input_shape)

        graph = helper.make_graph(
            nodes,
            "test_batchnormalization_8_9",
            [x, scale, B, mean, var],
            [y]
        )

        converted_model = self._converted(graph, helper.make_operatorsetid("", from_opset), to_opset)

        assert converted_model.graph.node[0].op_type == "BatchNormalization"
        assert converted_model.opset_import[0].version == to_opset

    # Test BatchNormalization Adapter: 9 -> 8
    def test_batchnormalization_9_8(self):  # type: () -> None
        from_opset = 9
        to_opset = 8
        data_type = TensorProto.FLOAT

        nodes = [onnx.helper.make_node(
            'BatchNormalization',
            inputs=['X', 'scale', 'B', 'mean', 'var'],
            outputs=['Y'],
        )]

        input_shape = (2, 3, 4, 5)
        x = onnx.helper.make_tensor_value_info("X", data_type, input_shape)
        scale = onnx.helper.make_tensor_value_info("scale", data_type, [input_shape[1]])
        B = onnx.helper.make_tensor_value_info("B", data_type, [input_shape[1]])
        mean = onnx.helper.make_tensor_value_info("mean", data_type, [input_shape[1]])
        var = onnx.helper.make_tensor_value_info("var", data_type, [input_shape[1]])
        y = onnx.helper.make_tensor_value_info("Y", data_type, input_shape)

        graph = onnx.helper.make_graph(
            nodes, "test_batchnormalization", [x, scale, B, mean, var], [y]
        )

        converted_model = self._converted(graph, helper.make_operatorsetid("", from_opset), to_opset)

        assert converted_model.graph.node[0].op_type == "BatchNormalization"
        assert converted_model.opset_import[0].version == to_opset

    # Test Constant Adapter: 8 -> 9
    def test_constant_8_9(self):  # type: () -> None
        from_opset = 8
        to_opset = 9
        data_type = TensorProto.FLOAT

        output_shape = [2, 3, 4]
        output_value = np.arange(24)

        nodes = [helper.make_node(
            "Constant",
            inputs=[],
            outputs=["Y"],
            value=helper.make_tensor("", data_type, output_shape, output_value))]

        graph = helper.make_graph(
            nodes,
            "test_constant",
            [],
            [onnx.helper.make_tensor_value_info("Y", data_type, output_shape)])

        converted_model = self._converted(graph, helper.make_operatorsetid("", from_opset), to_opset)

        assert converted_model.graph.node[0].op_type == "Constant"
        assert converted_model.graph.output[0].type.tensor_type.elem_type == data_type
        assert converted_model.opset_import[0].version == to_opset

    # Test Constant Adapter: 9 -> 8
    def test_constant_9_8(self):  # type: () -> None
        from_opset = 9
        to_opset = 8
        data_type = TensorProto.UINT64

        output_shape = [2, 3, 4]
        output_value = np.arange(24)

        nodes = [helper.make_node(
            "Constant",
            inputs=[],
            outputs=["Y"],
            value=helper.make_tensor("", data_type, output_shape, output_value))]

        graph = helper.make_graph(
            nodes,
            "test_constant",
            [],
            [onnx.helper.make_tensor_value_info("Y", data_type, output_shape)])

        converted_model = self._converted(graph, helper.make_operatorsetid("", from_opset), to_opset)

        assert converted_model.graph.node[0].op_type == "Constant"
        assert converted_model.graph.output[0].type.tensor_type.elem_type == data_type
        assert converted_model.opset_import[0].version == to_opset

    # Test Flatten Adapter: 8 -> 9
    def test_flatten_8_9(self):  # type: () -> None
        from_opset = 8
        to_opset = 9
        data_type = TensorProto.FLOAT

        nodes = [onnx.helper.make_node(
            "Flatten",
            inputs=["X"],
            outputs=["Y"],
            axis=1
        )]

        graph = helper.make_graph(
            nodes,
            "test_flatten",
            [onnx.helper.make_tensor_value_info("X", data_type, [2, 3, 4])],
            [onnx.helper.make_tensor_value_info("Y", data_type, [2, 12])])

        converted_model = self._converted(graph, helper.make_operatorsetid("", from_opset), to_opset)

        assert converted_model.graph.node[0].op_type == "Flatten"
        assert converted_model.graph.output[0].type.tensor_type.elem_type == data_type
        assert converted_model.opset_import[0].version == to_opset

    # Test Flatten Adapter: 9 -> 8
    def test_flatten_9_8(self):  # type: () -> None
        from_opset = 9
        to_opset = 8
        data_type = TensorProto.UINT64

        nodes = [onnx.helper.make_node(
            "Flatten",
            inputs=["X"],
            outputs=["Y"],
            axis=1
        )]

        graph = helper.make_graph(
            nodes,
            "test_flatten",
            [onnx.helper.make_tensor_value_info("X", data_type, [2, 3, 4])],
            [onnx.helper.make_tensor_value_info("Y", data_type, [2, 12])])

        converted_model = self._converted(graph, helper.make_operatorsetid("", from_opset), to_opset)

        assert converted_model.graph.node[1].op_type == "Flatten"
        assert converted_model.graph.output[0].type.tensor_type.elem_type == data_type
        assert converted_model.opset_import[0].version == to_opset

    # Test PRelu Adapter: 8 -> 9
    def test_prelu_8_9(self):  # type: () -> None
        from_opset = 8
        to_opset = 9
        data_type = TensorProto.FLOAT

        nodes = [onnx.helper.make_node(
            "PRelu",
            inputs=["X", "Slope"],
            outputs=["Y"]
        )]

        input_shape = [2, 3, 4]
        graph = helper.make_graph(
            nodes,
            "test_prelu",
            [onnx.helper.make_tensor_value_info("X", data_type, input_shape),
             onnx.helper.make_tensor_value_info("Slope", data_type, input_shape)],
            [onnx.helper.make_tensor_value_info("Y", data_type, input_shape)])

        converted_model = self._converted(graph, helper.make_operatorsetid("", from_opset), to_opset)

        assert converted_model.graph.node[0].op_type == "PRelu"
        assert converted_model.graph.output[0].type.tensor_type.elem_type == data_type
        assert converted_model.opset_import[0].version == to_opset

    # Test PRelu Adapter: 9 -> 8
    def test_prelu_9_8(self):  # type: () -> None
        from_opset = 9
        to_opset = 8
        data_type = TensorProto.UINT64

        nodes = [onnx.helper.make_node(
            "PRelu",
            inputs=["X", "Slope"],
            outputs=["Y"]
        )]

        input_shape = [2, 3, 4]
        graph = helper.make_graph(
            nodes,
            "test_prelu",
            [onnx.helper.make_tensor_value_info("X", data_type, input_shape),
             onnx.helper.make_tensor_value_info("Slope", data_type, input_shape)],
            [onnx.helper.make_tensor_value_info("Y", data_type, input_shape)])

        converted_model = self._converted(graph, helper.make_operatorsetid("", from_opset), to_opset)

        assert converted_model.graph.node[2].op_type == "PRelu"
        assert converted_model.graph.output[0].type.tensor_type.elem_type == data_type
        assert converted_model.opset_import[0].version == to_opset

    # Test Greater Adapter: 8 -> 9
    def test_greater_8_9(self):  # type: () -> None
        from_opset = 8
        to_opset = 9
        data_type = TensorProto.FLOAT

        nodes = [onnx.helper.make_node(
            "Greater",
            inputs=["X1", "X2"],
            outputs=["Y"]
        )]

        input_shape = [2, 3, 4]
        graph = helper.make_graph(
            nodes,
            "test_greater",
            [onnx.helper.make_tensor_value_info("X1", data_type, input_shape),
             onnx.helper.make_tensor_value_info("X2", data_type, input_shape)],
            [onnx.helper.make_tensor_value_info("Y", TensorProto.BOOL, input_shape)])

        converted_model = self._converted(graph, helper.make_operatorsetid("", from_opset), to_opset)

        assert converted_model.graph.node[0].op_type == "Greater"
        assert converted_model.graph.output[0].type.tensor_type.elem_type == TensorProto.BOOL
        assert converted_model.opset_import[0].version == to_opset

    # Test Greater Adapter: 9 -> 8
    def test_greater_9_8(self):  # type: () -> None
        from_opset = 9
        to_opset = 8
        data_type = TensorProto.UINT64

        nodes = [onnx.helper.make_node(
            "Greater",
            inputs=["X1", "X2"],
            outputs=["Y"]
        )]

        input_shape = [2, 3, 4]
        graph = helper.make_graph(
            nodes,
            "test_greater",
            [onnx.helper.make_tensor_value_info("X1", data_type, input_shape),
             onnx.helper.make_tensor_value_info("X2", data_type, input_shape)],
            [onnx.helper.make_tensor_value_info("Y", TensorProto.BOOL, input_shape)])

        converted_model = self._converted(graph, helper.make_operatorsetid("", from_opset), to_opset)

        assert converted_model.graph.node[2].op_type == "Greater"
        assert converted_model.graph.output[0].type.tensor_type.elem_type == TensorProto.BOOL
        assert converted_model.opset_import[0].version == to_opset

    # Test Less Adapter: 8 -> 9
    def test_less_8_9(self):  # type: () -> None
        from_opset = 8
        to_opset = 9
        data_type = TensorProto.FLOAT

        nodes = [onnx.helper.make_node(
            "Less",
            inputs=["X1", "X2"],
            outputs=["Y"]
        )]

        input_shape = [2, 3, 4]
        graph = helper.make_graph(
            nodes,
            "test_less",
            [onnx.helper.make_tensor_value_info("X1", data_type, input_shape),
             onnx.helper.make_tensor_value_info("X2", data_type, input_shape)],
            [onnx.helper.make_tensor_value_info("Y", TensorProto.BOOL, input_shape)])

        converted_model = self._converted(graph, helper.make_operatorsetid("", from_opset), to_opset)

        assert converted_model.graph.node[0].op_type == "Less"
        assert converted_model.graph.output[0].type.tensor_type.elem_type == TensorProto.BOOL
        assert converted_model.opset_import[0].version == to_opset

    # Test Less Adapter: 9 -> 8
    def test_less_9_8(self):  # type: () -> None
        from_opset = 9
        to_opset = 8
        data_type = TensorProto.UINT64

        nodes = [onnx.helper.make_node(
            "Less",
            inputs=["X1", "X2"],
            outputs=["Y"]
        )]

        input_shape = [2, 3, 4]
        graph = helper.make_graph(
            nodes,
            "test_less",
            [onnx.helper.make_tensor_value_info("X1", data_type, input_shape),
             onnx.helper.make_tensor_value_info("X2", data_type, input_shape)],
            [onnx.helper.make_tensor_value_info("Y", TensorProto.BOOL, input_shape)])

        converted_model = self._converted(graph, helper.make_operatorsetid("", from_opset), to_opset)

        assert converted_model.graph.node[2].op_type == "Less"
        assert converted_model.graph.output[0].type.tensor_type.elem_type == TensorProto.BOOL
        assert converted_model.opset_import[0].version == to_opset

    # Test MatMul Adapter: 8 -> 9
    def test_matmul_8_9(self):  # type: () -> None
        from_opset = 8
        to_opset = 9
        data_type = TensorProto.FLOAT

        nodes = [onnx.helper.make_node(
            "MatMul",
            inputs=["X1", "X2"],
            outputs=["Y"]
        )]

        graph = helper.make_graph(
            nodes,
            "test_matmul",
            [onnx.helper.make_tensor_value_info("X1", data_type, [3, 4]),
             onnx.helper.make_tensor_value_info("X2", data_type, [4, 3])],
            [onnx.helper.make_tensor_value_info("Y", data_type, [3, 3])])

        converted_model = self._converted(graph, helper.make_operatorsetid("", from_opset), to_opset)

        assert converted_model.graph.node[0].op_type == "MatMul"
        assert converted_model.graph.output[0].type.tensor_type.elem_type == data_type
        assert converted_model.opset_import[0].version == to_opset

    # Test MatMul Adapter: 9 -> 8
    def test_matmul_9_8(self):  # type: () -> None
        from_opset = 9
        to_opset = 8
        data_type = TensorProto.UINT64

        nodes = [onnx.helper.make_node(
            "MatMul",
            inputs=["X1", "X2"],
            outputs=["Y"]
        )]

        graph = helper.make_graph(
            nodes,
            "test_matmul",
            [onnx.helper.make_tensor_value_info("X1", data_type, [3, 4]),
             onnx.helper.make_tensor_value_info("X2", data_type, [4, 3])],
            [onnx.helper.make_tensor_value_info("Y", data_type, [3, 3])])

        converted_model = self._converted(graph, helper.make_operatorsetid("", from_opset), to_opset)

        assert converted_model.graph.node[2].op_type == "MatMul"
        assert converted_model.graph.output[0].type.tensor_type.elem_type == data_type
        assert converted_model.opset_import[0].version == to_opset

    # Test Gemm Adapter: 8 -> 9
    def test_gemm_8_9(self):  # type: () -> None
        from_opset = 8
        to_opset = 9
        data_type = TensorProto.FLOAT

        nodes = [onnx.helper.make_node(
            "Gemm",
            inputs=["X1", "X2", "X3"],
            outputs=["Y"]
        )]

        graph = helper.make_graph(
            nodes,
            "test_gemm",
            [onnx.helper.make_tensor_value_info("X1", data_type, [3, 4]),
             onnx.helper.make_tensor_value_info("X2", data_type, [4, 3]),
             onnx.helper.make_tensor_value_info("X3", data_type, [3, 3])],
            [onnx.helper.make_tensor_value_info("Y", data_type, [3, 3])])

        converted_model = self._converted(graph, helper.make_operatorsetid("", from_opset), to_opset)

        assert converted_model.graph.node[0].op_type == "Gemm"
        assert converted_model.graph.output[0].type.tensor_type.elem_type == data_type
        assert converted_model.opset_import[0].version == to_opset

    # Test Gemm Adapter: 9 -> 8
    def test_gemm_9_8(self):  # type: () -> None
        from_opset = 9
        to_opset = 8
        data_type = TensorProto.UINT64

        nodes = [onnx.helper.make_node(
            "Gemm",
            inputs=["X1", "X2", "X3"],
            outputs=["Y"]
        )]

        graph = helper.make_graph(
            nodes,
            "test_gemm",
            [onnx.helper.make_tensor_value_info("X1", data_type, [3, 4]),
             onnx.helper.make_tensor_value_info("X2", data_type, [4, 3]),
             onnx.helper.make_tensor_value_info("X3", data_type, [3, 3])],
            [onnx.helper.make_tensor_value_info("Y", data_type, [3, 3])])

        converted_model = self._converted(graph, helper.make_operatorsetid("", from_opset), to_opset)

        assert converted_model.graph.node[3].op_type == "Gemm"
        assert converted_model.graph.output[0].type.tensor_type.elem_type == data_type
        assert converted_model.opset_import[0].version == to_opset

    # Test Upsample Adapter: 8 -> 9
    def test_upsample_8_9(self):  # type: () -> None
        from_opset = 8
        to_opset = 9
        data_type = TensorProto.FLOAT

        nodes = [onnx.helper.make_node(
            "Upsample",
            inputs=["X"],
            outputs=["Y"],
            mode="nearest",
            scales=[1.0, 1.0, 2.0, 3.0],
        )]

        graph = helper.make_graph(
            nodes,
            "test_upsample_8_9",
            [onnx.helper.make_tensor_value_info("X", data_type, [1, 1, 2, 2])],
            [onnx.helper.make_tensor_value_info("Y", data_type, [1, 1, 4, 6])]
        )

        converted_model = self._converted(graph, helper.make_operatorsetid("", from_opset), to_opset)

        assert len(converted_model.graph.node) == 1
        assert converted_model.graph.node[0].op_type == "Upsample"
        assert len(converted_model.graph.node[0].attribute) == 1
        assert converted_model.graph.node[0].attribute[0].name == "mode"
        assert converted_model.opset_import[0].version == to_opset

    # Test Helper for Upsample Adapter: 9 -> 8
    def helper_upsample_with_initializer(self, raw_scale=False):  # type: (bool) -> None
        from_opset = 9
        to_opset = 8
        data_type = TensorProto.FLOAT

        nodes = [onnx.helper.make_node(
            "Upsample",
            inputs=["X", "Scales"],
            outputs=["Y"],
            mode="nearest"
        )]

        scale_value = [1.0, 1.0, 2.0, 3.0]
        scale_tensor = onnx.helper.make_tensor("Scales", onnx.TensorProto.FLOAT, [4], bytes(struct.pack("4f", *scale_value)) if raw_scale else scale_value, raw_scale)

        graph = helper.make_graph(
            nodes,
            "test_upsample",
            [onnx.helper.make_tensor_value_info("X", data_type, [1, 1, 2, 2]),
             onnx.helper.make_tensor_value_info("Scales", data_type, [4])],
            [onnx.helper.make_tensor_value_info("Y", data_type, [1, 1, 4, 6])],
            [scale_tensor])

        converted_model = self._converted(graph, helper.make_operatorsetid("", from_opset), to_opset)

        assert converted_model.graph.node[0].op_type == "Upsample"
        assert len(converted_model.graph.initializer) == 0
        assert len(converted_model.graph.node[0].attribute) == 2
        assert converted_model.graph.node[0].attribute[1].name == "scales"
        assert converted_model.opset_import[0].version == to_opset

    # Test Helper for Upsample Adapter: 9 -> 8
    def helper_upsample_with_constant(self, raw_scale=False):  # type: (bool) -> None
        from_opset = 9
        to_opset = 8
        data_type = TensorProto.FLOAT

        scale_value = [1.0, 1.0, 2.0, 3.0]
        scale_tensor = onnx.helper.make_tensor("const_value", onnx.TensorProto.FLOAT, [4], bytes(struct.pack("4f", *scale_value)) if raw_scale else scale_value, raw_scale)
        nodes = [
            onnx.helper.make_node(
                'Constant',
                inputs=[],
                outputs=['Constant_Output'],
                value=scale_tensor),
            onnx.helper.make_node(
                "Upsample",
                inputs=["X", "Constant_Output"],
                outputs=["Y"],
                mode="nearest")]

        graph = helper.make_graph(
            nodes,
            "test_upsample",
            [onnx.helper.make_tensor_value_info("X", data_type, [1, 1, 2, 2])],
            [onnx.helper.make_tensor_value_info("Y", data_type, [1, 1, 4, 6])],
            value_info=[onnx.helper.make_tensor_value_info("Constant_Output", data_type, [4])])

        converted_model = self._converted(graph, helper.make_operatorsetid("", from_opset), to_opset)

        assert len(converted_model.graph.node) == 1
        assert converted_model.graph.node[0].op_type == "Upsample"
        assert len(converted_model.graph.node[0].attribute) == 2
        assert converted_model.graph.node[0].attribute[1].name == "scales"
        assert converted_model.opset_import[0].version == to_opset

    # Test Upsample Adapter: 9 -> 8
    def test_upsample_with_constant_node_9_8(self):  # type: () -> None
        self.helper_upsample_with_constant(raw_scale=False)

    # Test Upsample Adapter: 9 -> 8
    def test_upsample_with_initializer_9_8(self):  # type: () -> None
        self.helper_upsample_with_initializer(raw_scale=False)

    # Test Upsample Adapter: 9 -> 8
    def test_upsample_with_raw_initializer_9_8(self):  # type: () -> None
        self.helper_upsample_with_constant(raw_scale=True)

    # Test Upsample Adapter: 9 -> 8
    def test_upsample_with_raw_constant_node_9_8(self):  # type: () -> None
        self.helper_upsample_with_constant(raw_scale=True)

    # Test Scan Adapter: 8 -> 9
    def test_scan_8_9(self):  # type: () -> None
        from_opset = 8
        to_opset = 9
        data_type = TensorProto.FLOAT

        node1 = onnx.helper.make_node("Add", inputs=["sum_in", "next"], outputs=["sum_out"],)
        node2 = onnx.helper.make_node("Identity", inputs=["sum_out"], outputs=["scan_out"],)
        g = onnx.helper.make_graph(
            [node1, node2],
            "scan_body",
            [onnx.helper.make_tensor_value_info("sum_in", data_type, [2]),
             onnx.helper.make_tensor_value_info("next", data_type, [2])],
            [onnx.helper.make_tensor_value_info("sum_out", data_type, [2]),
             onnx.helper.make_tensor_value_info("scan_out", data_type, [2])]
        )

        nodes = [onnx.helper.make_node(
            "Scan",
            inputs=["", "initial", "x"],
            outputs=["y", "z"],
            body=g,
            num_scan_inputs=1,
        )]

        seq_lens = onnx.helper.make_empty_tensor_value_info(" ")
        initial = onnx.helper.make_tensor_value_info("initial", data_type, [1, 2])
        x = onnx.helper.make_tensor_value_info("x", data_type, [1, 3, 2])
        y = onnx.helper.make_tensor_value_info("y", data_type, [1, 2])
        z = onnx.helper.make_tensor_value_info("z", data_type, [1, 3, 2])

        graph = onnx.helper.make_graph(
            nodes, "test_scan_8_9", [seq_lens, initial, x], [y, z]
        )

        converted_model = self._converted(graph, helper.make_operatorsetid("", from_opset), to_opset)

        assert converted_model.graph.node[0].op_type == "Scan"
        assert converted_model.opset_import[0].version == to_opset

    # Test Cast Adapter: 8 -> 9
    def test_cast_8_9(self):  # type: () -> None
        from_opset = 8
        to_opset = 9
        data_type_from = TensorProto.FLOAT
        data_type_to = TensorProto.UINT32

        nodes = [onnx.helper.make_node(
            "Cast",
            inputs=["X"],
            outputs=["Y"],
            to=TensorProto.UINT32
        )]

        graph = helper.make_graph(
            nodes,
            "test_cast",
            [onnx.helper.make_tensor_value_info("X", data_type_from, [2, 3])],
            [onnx.helper.make_tensor_value_info("Y", data_type_to, [2, 3])])

        converted_model = self._converted(graph, helper.make_operatorsetid("", from_opset), to_opset)

        assert converted_model.graph.node[0].op_type == "Cast"
        assert converted_model.graph.output[0].type.tensor_type.elem_type == data_type_to
        assert converted_model.opset_import[0].version == to_opset


if __name__ == '__main__':
    unittest.main()
