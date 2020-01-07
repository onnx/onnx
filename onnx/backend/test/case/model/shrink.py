from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class ShrinkTest(Base):

    @staticmethod
    def export():  # type: () -> None

        node = onnx.helper.make_node(
            'Shrink', ['x'], ['y'], lambd=1.5, bias=1.5,)
        graph = onnx.helper.make_graph(
            nodes=[node],
            name='Shrink',
            inputs=[onnx.helper.make_tensor_value_info(
                'x', onnx.TensorProto.FLOAT, [5])],
            outputs=[onnx.helper.make_tensor_value_info(
                'y', onnx.TensorProto.FLOAT, [5])])
        model = onnx.helper.make_model(graph,
                                       producer_name='backend-test')

        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        y = np.array([-0.5, 0.0, 0.0, 0.0, 0.5], dtype=np.float32)

        expect(model, inputs=[x], outputs=[y],
               name='test_shrink')
