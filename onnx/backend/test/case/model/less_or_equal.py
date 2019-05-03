from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect
from typing import Sequence


class LessOrEqual(Base):

    @staticmethod
    def export():  # type: () -> None
        node = onnx.helper.make_node(
            'LessOrEqual',
            ['x', 'y'],
            ['less_or_equal'],
            name='test')

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.random.randn(3, 4, 5).astype(np.float32)
        z = np.less_equal(x, y)

        graph = onnx.helper.make_graph(
            nodes=[node],
            name='LessOrEqual',
            inputs=[onnx.helper.make_tensor_value_info('x',
                                                       onnx.TensorProto.FLOAT,
                                                       x.shape),
                    onnx.helper.make_tensor_value_info('y',
                                                       onnx.TensorProto.FLOAT,
                                                       y.shape)],
            outputs=[onnx.helper.make_tensor_value_info('z',
                                                        onnx.TensorProto.FLOAT,
                                                        z.shape)])
        model = onnx.helper.make_model(graph, producer_name='backend-test')
        expect(model, inputs=[x], outputs=[y],
               name='test_less_or_equal_model')
