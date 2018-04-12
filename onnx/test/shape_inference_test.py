from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from onnx import checker, helper, TensorProto

import onnx.shape_inference
import unittest


class TestShapeInference(unittest.TestCase):

    def _inferred(self, graph):
        orig_model = helper.make_model(graph, producer_name='onnx-test')
        inferred_model = onnx.shape_inference.infer_shapes(orig_model)
        checker.check_model(inferred_model)
        return inferred_model

    def test_transpose(self):
        trans = helper.make_node("Transpose", ["X"], ["Y"], perm=[1, 0, 2])
        graph = helper.make_graph(
            [trans],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (2, 3, 4))],
            [])

        inferred_model = self._inferred(graph)

        vis_expected = [
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, (3, 2, 4))
        ]

        assert list(inferred_model.graph.value_info) == vis_expected, \
            inferred_model.graph.value_info

    def test_transpose_preexisting(self):
        trans = helper.make_node("Transpose", ["X"], ["Y"], perm=[1, 0, 2])
        graph = helper.make_graph(
            [trans],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (2, 3, 4))],
            [])
        graph.value_info.extend([
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
        ])

        inferred_model = self._inferred(graph)

        vis_expected = [
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, (3, 2, 4))
        ]

        assert list(inferred_model.graph.value_info) == vis_expected, \
            inferred_model.graph.value_info

    def test_transpose_partial(self):
        trans = helper.make_node("Transpose", ["X"], ["Y"], perm=[1, 0, 2])
        graph = helper.make_graph(
            [trans],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (2, 3, 4))],
            [])
        graph.value_info.extend([
            helper.make_tensor_value_info("Y", TensorProto.UNDEFINED, (3, "a", "b"))
        ])

        inferred_model = self._inferred(graph)

        vis_expected = [
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, (3, 2, 4))
        ]

        assert list(inferred_model.graph.value_info) == vis_expected, \
            inferred_model.graph.value_info

    def test_transpose_preexisting_incorrect_shape(self):
        trans = helper.make_node("Transpose", ["X"], ["Y"], perm=[1, 0, 2])
        graph = helper.make_graph(
            [trans],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (2, 3, 4))],
            [])
        graph.value_info.extend([
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, (5, 5, 5))
        ])

        self.assertRaises(RuntimeError, self._inferred, graph)

    def test_transpose_preexisting_incorrect_type(self):
        trans = helper.make_node("Transpose", ["X"], ["Y"], perm=[1, 0, 2])
        graph = helper.make_graph(
            [trans],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (2, 3, 4))],
            [])
        graph.value_info.extend([
            helper.make_tensor_value_info("Y", TensorProto.STRING, (3, 2, 4))
        ])

        self.assertRaises(RuntimeError, self._inferred, graph)


if __name__ == '__main__':
    unittest.main()
