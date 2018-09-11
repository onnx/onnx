from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import unittest

import numpy as np  # type: ignore

from onnx import checker, helper
from onnx import TensorProto


class TestChecker(unittest.TestCase):
    @property
    def _sample_float_tensor(self):  # type: () -> TensorProto
        np_array = np.random.randn(2, 3).astype(np.float32)
        return helper.make_tensor(
            name='test',
            data_type=TensorProto.FLOAT,
            dims=(2, 3),
            vals=np_array.reshape(6).tolist()
        )

    def test_check_node(self):  # type: () -> None
        node = helper.make_node(
            "Relu", ["X"], ["Y"], name="test")

        checker.check_node(node)

    def test_check_node_input_marked_optional(self):  # type: () -> None
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
        self.assertRaises(checker.ValidationError, checker.check_node, node)

    def test_check_graph(self):  # type: () -> None
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

    def test_check_graph_optional_input(self):  # type: () -> None
        node = helper.make_node(
            "ConstantFill", [""], ["Y"], name="test")
        graph = helper.make_graph(
            [node],
            "test",
            [],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2])])
        checker.check_graph(graph)

    def test_check_graph_ssa(self):  # type: () -> None
        relu1 = helper.make_node(
            "Relu", ["X"], ["Z"], name="relu1")
        relu2 = helper.make_node(
            "Relu", ["Y"], ["Z"], name="relu2")

        graph = helper.make_graph(
            [relu1, relu2],
            "test",
            inputs=[
                helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2]),
                helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2])
            ],
            outputs=[
                helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1, 2])
            ]
        )
        self.assertRaises(checker.ValidationError, checker.check_graph, graph)

    def test_check_graph_topologically_sorted(self):  # type: () -> None
        n1 = helper.make_node(
            "Scale", ["X"], ["Y"], scale=2., name="n1")
        n2 = helper.make_node(
            "Scale", ["Y"], ["Z"], scale=3., name="n2")

        graph = helper.make_graph(
            [n2, n1],
            "test",
            inputs=[
                helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])
            ],
            outputs=[
                helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1, 2])
            ]
        )
        self.assertRaises(checker.ValidationError, checker.check_graph, graph)

    def test_check_model(self):  # type: () -> None
        node = helper.make_node(
            "Relu", ["X"], ["Y"], name="test")
        graph = helper.make_graph(
            [node],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2])])
        model = helper.make_model(graph, producer_name='test')

        checker.check_model(model)

    def test_check_old_model(self):  # type: () -> None
        node = helper.make_node(
            "Pad", ["X"], ["Y"], paddings=(0, 0, 0, 0))
        graph = helper.make_graph(
            [node],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2])])
        onnx_id = helper.make_opsetid("", 1)
        model = helper.make_model(graph, producer_name='test', opset_imports=[onnx_id])

        checker.check_model(model)

    def test_check_tensor(self):  # type: () -> None
        tensor = self._sample_float_tensor
        checker.check_tensor(tensor)

        tensor.raw_data = np.random.randn(2, 3).astype(np.float32).tobytes()
        self.assertRaises(checker.ValidationError, checker.check_tensor, tensor)

    def test_check_string_tensor(self):  # type: () -> None
        tensor = TensorProto()
        tensor.data_type = TensorProto.STRING
        tensor.dims.append(1)
        tensor.string_data.append('Test'.encode('utf-8'))
        checker.check_tensor(tensor)

        del tensor.string_data[:]
        tensor.raw_data = 'Test'.encode('utf-8')
        # string data should not be stored in raw_data field
        self.assertRaises(checker.ValidationError, checker.check_tensor, tensor)

    def test_check_tensor_mismatched_field(self):  # type: () -> None
        tensor = self._sample_float_tensor
        tensor.data_type = TensorProto.INT32
        self.assertRaises(checker.ValidationError, checker.check_tensor, tensor)

    def test_nested_graph(self):  # type: () -> None
        n1 = helper.make_node(
            "Scale", ["X"], ["Y"], scale=2., name="n1")
        n2 = helper.make_node(
            "Scale", ["Y"], ["Z"], scale=3., name="n2")

        graph = helper.make_graph(
            [n1, n2],
            "nested",
            inputs=[
                helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])
            ],
            outputs=[
                helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1, 2])
            ]
        )

        i1 = helper.make_node(
            "If", ["cond"], ["Z"], then_branch=graph, else_branch=graph)

        graph = helper.make_graph(
            [i1],
            "test",
            inputs=[
                helper.make_tensor_value_info("cond", TensorProto.BOOL, [1])
            ],
            outputs=[],
        )

        checker.check_graph(graph)
        #self.assertRaises(checker.ValidationError, checker.check_graph, graph)

    @property
    def _sample_0_elem_tensor(self):  # type: () -> TensorProto
        np_array = np.random.randn(0, 3).astype(np.float32)
        return helper.make_tensor(
            name='test',
            data_type=TensorProto.FLOAT,
            dims=(0, 3),
            vals=np_array.reshape(0).tolist()
        )

    def test_check_tensor_zero_elem(self):  # type: () -> None
        tensor = self._sample_0_elem_tensor
        checker.check_tensor(tensor)


if __name__ == '__main__':
    unittest.main()
