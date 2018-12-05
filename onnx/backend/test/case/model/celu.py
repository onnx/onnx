from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class SingleCelu(Base):

    @staticmethod
    def export():  # type: () -> None

        node = onnx.helper.make_node(
            'Celu', ['x'], ['y'], name='test')
        graph = onnx.helper.make_graph(
            nodes=[node],
            name='SingleCelu',
            inputs=[onnx.helper.make_tensor_value_info(
                'x', onnx.TensorProto.FLOAT, [1, 2])],
            outputs=[onnx.helper.make_tensor_value_info(
                'y', onnx.TensorProto.FLOAT, [1, 2])])
        model = onnx.helper.make_model(graph, producer_name='backend-test')

        x = np.array([2, -3, 8, -2, -1], dtype=np.float32)
        y = np.array([2, -0.9602, 8, -0.8647, -0.6321], dtype=np.float32)

        expect(model, inputs=[x], outputs=[y],
               name='test_single_celu_model')
