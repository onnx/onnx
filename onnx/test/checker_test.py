from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import unittest

import numpy as np

from onnx import checker, helper
from onnx.onnx_pb2 import TensorProto


class TestChecker(unittest.TestCase):
    @property
    def _sample_float_tensor(self):
        np_array = np.random.randn(2, 3).astype(np.float32)
        return helper.make_tensor(
            name='test',
            data_type=TensorProto.FLOAT,
            dims=(2, 3),
            vals=np_array.reshape(6).tolist()
        )

    def test_check_node(self):
        node = helper.make_node(
            "Relu", ["X"], ["Y"], name="test")

        checker.check_node(node)

    def test_check_node_input_marked_optional(self):
        # Constant fill's input is marked optional
        node = helper.make_node(
            "ConstantFill", [], ["Y"], name="test")
        checker.check_node(node)

        # Explicitly pass the empty string as optional
        node = helper.make_node(
            "ConstantFill", [""], ["Y"], name="test")

        # Input of RELU is not optional
        node = helper.make_node(
            "Relu", [""], ["Y"], name="test")
        self.assertRaises(ValueError, checker.check_node, node)

    def test_check_graph(self):
        node = helper.make_node(
            "Relu", ["X"], ["Y"], name="test")
        graph = helper.make_graph(
            [node],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2])])
        checker.check_graph(graph)

        graph.initializer.extend([self._sample_float_tensor])

        graph.initializer[0].name = 'no-exist'
        self.assertRaises(checker.ValidationError, checker.check_graph, graph)

        graph.initializer[0].name = 'X'
        checker.check_graph(graph)

    def test_check_graph_optional_input(self):
        node = helper.make_node(
            "ConstantFill", [""], ["Y"], name="test")
        graph = helper.make_graph(
            [node],
            "test",
            [],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2])])
        checker.check_graph(graph)

    def test_check_model(self):
        node = helper.make_node(
            "Relu", ["X"], ["Y"], name="test")
        graph = helper.make_graph(
            [node],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2])])
        model = helper.make_model(graph, producer_name='test')

        checker.check_model(model)

    def test_check_tensor(self):
        tensor = self._sample_float_tensor
        checker.check_tensor(tensor)

        tensor.raw_data = np.random.randn(2, 3).astype(np.float32).tobytes()
        self.assertRaises(checker.ValidationError, checker.check_tensor, tensor)

    def test_check_string_tensor(self):
        tensor = TensorProto()
        tensor.data_type = TensorProto.STRING
        tensor.dims.append(1)
        tensor.string_data.append('Test'.encode('utf-8'))
        checker.check_tensor(tensor)

        del tensor.string_data[:]
        tensor.raw_data = 'Test'.encode('utf-8')
        # string data should not be stored in raw_data field
        self.assertRaises(checker.ValidationError, checker.check_tensor, tensor)

    def test_check_tensor_mismatched_field(self):
        tensor = self._sample_float_tensor
        tensor.data_type = TensorProto.INT32
        self.assertRaises(checker.ValidationError, checker.check_tensor, tensor)


if __name__ == '__main__':
    unittest.main()
