from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import onnx.utils
from onnx import helper, TensorProto
import numpy as np

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

    def test_update_inputs_outputs_dim(self):  # type: () -> None
        node_def = helper.make_node(
            "Conv",
            inputs=['x', 'W'],
            outputs=['y'],
            kernel_shape=[3, 3],
            strides=[2, 2],
        )
        graph_def = helper.make_graph(
            [node_def],
            'test',
            [helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1, 5, 5]),
             helper.make_tensor_value_info('W', TensorProto.FLOAT, [1, 1, 3, 3])],
            [helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 1, 2, 2])]
        )
        model_def = helper.make_model(graph_def, producer_name='test')
        updated_def = onnx.utils.update_inputs_outputs_dims(model_def, [[1, 1, 'x1', -1], [1, 1, 3, 3]], [[1, 1, -1, -1]])
        onnx.checker.check_model(updated_def)
        self.assertEqual(updated_def.graph.input[0].type.tensor_type.shape.dim[2].dim_param, 'x1')
        self.assertEqual(updated_def.graph.input[0].type.tensor_type.shape.dim[3].dim_param, 'in_0_3')
        self.assertEqual(updated_def.graph.output[0].type.tensor_type.shape.dim[2].dim_param, 'out_0_2')
        self.assertEqual(updated_def.graph.output[0].type.tensor_type.shape.dim[3].dim_param, 'out_0_3')

if __name__ == '__main__':
    unittest.main()
