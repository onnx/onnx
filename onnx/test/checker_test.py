import unittest

import onnx
from onnx import checker, helper


class TestChecker(unittest.TestCase):
    def test_check_node(self):
        node = helper.make_node(
            "Relu", ["X"], ["Y"], name="test")

        checker.check_node(node)

    def test_check_graph(self):
        node = helper.make_node(
            "Relu", ["X"], ["Y"], name="test")
        graph = helper.make_graph(
            [node],
            "test",
            [helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 2])],
            [helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 2])])

        checker.check_graph(graph)

    def test_check_model(self):
        node = helper.make_node(
            "Relu", ["X"], ["Y"], name="test")
        graph = helper.make_graph(
            [node],
            "test",
            [helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 2])],
            [helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 2])])
        model = helper.make_model(graph, producer_name='test')

        checker.check_model(model)
