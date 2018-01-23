from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import onnx
from ..base import Base
from . import expect


class SingleRelu(Base):

    @staticmethod
    def export():

        node = onnx.helper.make_node(
            'Relu', ['x'], ['y'], name='test')
        graph = onnx.helper.make_graph(
            nodes=[node],
            name='SingleRelu',
            inputs=[onnx.helper.make_tensor_value_info(
                'x', onnx.TensorProto.FLOAT, [1, 2])],
            outputs=[onnx.helper.make_tensor_value_info(
                'y', onnx.TensorProto.FLOAT, [1, 2])])
        model = onnx.helper.make_model(graph, producer_name='backend-test')

        x = np.random.randn(1, 2).astype(np.float32)
        y = np.maximum(x, 0)

        expect(model, inputs=[x], outputs=[y],
               name='test_single_relu_model')
