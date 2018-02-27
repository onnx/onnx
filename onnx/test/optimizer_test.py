from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from onnx import checker, helper, ModelProto, TensorProto

import onnx.optimizer
import unittest

class TestOptimizer(unittest.TestCase):

    def _optimized(self, graph, name):
        orig_model = helper.make_model(graph, producer_name='onnx-test')
        orig_model_str = orig_model.SerializeToString()
        optimized_model_str = onnx.optimizer.optimize(orig_model_str, [name])
        optimized_model = ModelProto()
        optimized_model.ParseFromString(optimized_model_str)
        checker.check_model(optimized_model)
        return optimized_model

    def test_nop_transpose(self):
        trans = helper.make_node("Transpose", ["X"], ["Y"], perm=[0,1])
        graph = helper.make_graph(
            [trans],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (2, 3))],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, (2, 3))])
        optimized_model = self._optimized(graph, "eliminate_nop_transpose")

        for node in optimized_model.graph.node:
            assert node.op_type != "Transpose"

    def test_nop_transpose_default(self):
        trans = helper.make_node("Transpose", ["X"], ["Y"])
        graph = helper.make_graph(
            [trans],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (2, 3))],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, (3, 2))])
        optimized_model = self._optimized(graph, "eliminate_nop_transpose")

        assert len(list(optimized_model.graph.node)) == 1
        assert optimized_model.graph.node[0].op_type == "Transpose"

    def test_fuse_transpose(self):
        trans1 = helper.make_node("Transpose", ["X"], ["Y"], perm=[1,0,2])
        trans2 = helper.make_node("Transpose", ["Y"], ["Z"], perm=[2,0,1])
        trans3 = helper.make_node("Transpose", ["Z"], ["A"], perm=[2,0,1])
        graph = helper.make_graph(
            [trans1, trans2, trans3],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (2, 3, 4))],
            [helper.make_tensor_value_info("A", TensorProto.FLOAT, (4, 3, 2))])
        optimized_model = self._optimized(graph, "fuse_consecutive_transposes")

        assert len(list(optimized_model.graph.node)) == 1

    def test_fuse_transpose_default(self):
        trans1 = helper.make_node("Transpose", ["X"], ["Y"])
        trans2 = helper.make_node("Transpose", ["Y"], ["Z"])
        graph = helper.make_graph(
            [trans1, trans2],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (2, 3, 4))],
            [helper.make_tensor_value_info("Z", TensorProto.FLOAT, (2, 3, 4))])
        optimized_model = self._optimized(graph, "fuse_consecutive_transposes")

        assert len(list(optimized_model.graph.node)) == 0

    def test_fuse_transpose_default_no_fuse(self):
        trans1 = helper.make_node("Transpose", ["X"], ["Y"])
        trans2 = helper.make_node("Transpose", ["Y"], ["Z"], perm=[0, 1, 2])
        graph = helper.make_graph(
            [trans1, trans2],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (2, 3, 4))],
            [helper.make_tensor_value_info("Z", TensorProto.FLOAT, (4, 3, 2))])
        optimized_model = self._optimized(graph, "fuse_consecutive_transposes")

        assert len(list(optimized_model.graph.node)) == 2
        for node in optimized_model.graph.node:
            assert node.op_type == "Transpose"

    def test_fuse_transpose_into_gemm(self):
        trans1 = helper.make_node("Transpose", ["X"], ["A"], perm=[1,0])
        trans2 = helper.make_node("Transpose", ["Y"], ["B"], perm=[1,0])
        gemm = helper.make_node("Gemm", ["A", "B", "C"], ["Z"])
        graph = helper.make_graph(
            [trans1, trans2, gemm],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (2, 3)),
             helper.make_tensor_value_info("Y", TensorProto.FLOAT, (5, 2)),
             helper.make_tensor_value_info("C", TensorProto.FLOAT, (3, 5))],
            [helper.make_tensor_value_info("Z", TensorProto.FLOAT, (3, 5))])
        optimized_model = self._optimized(graph, "fuse_transpose_into_gemm")

        assert len(list(optimized_model.graph.node)) == 1
        assert optimized_model.graph.node[0].op_type == "Gemm"


if __name__ == '__main__':
    unittest.main()
