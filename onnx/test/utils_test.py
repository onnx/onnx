from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import onnx.utils
from onnx import helper, TensorProto


class TestUtilityFunctions(unittest.TestCase):
    def test_polish_model(self):  # type: () -> None
        node_def = helper.make_node(
            "Relu", ["X"], ["Y"], doc_string="ABC")
        graph_def = helper.make_graph(
            [node_def],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2])])
        model_def = helper.make_model(graph_def, producer_name='test')
        polished_def = onnx.utils.polish_model(model_def)
        self.assertEqual(polished_def.producer_name, 'test')
        self.assertEqual(len(polished_def.graph.node), 1)
        self.assertFalse(polished_def.graph.node[0].HasField('doc_string'))


if __name__ == '__main__':
    unittest.main()
