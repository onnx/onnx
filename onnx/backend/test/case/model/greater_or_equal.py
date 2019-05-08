from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect
from typing import Sequence


class GreaterOrEqual(Base):

    @staticmethod
    def export():  # type: () -> None
        node = onnx.helper.make_node(
            'GreaterOrEqual',
            ['x', 'y'],
            ['greater_or_equal'],
            name='test')

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.random.randn(3, 4, 5).astype(np.float32)
        z = np.greater_equal(x, y)

        graph = onnx.helper.make_graph(
            nodes=[node],
            name='GreaterOrEqual',
            inputs=[onnx.helper.make_tensor_value_info('x',
                                                       onnx.TensorProto.FLOAT,
                                                       x.shape),
                    onnx.helper.make_tensor_value_info('y',
                                                       onnx.TensorProto.FLOAT,
                                                       y.shape)],
            outputs=[onnx.helper.make_tensor_value_info('z',
                                                        onnx.TensorProto.BOOL,
                                                        z.shape)])
        model = onnx.helper.make_model(graph, producer_name='backend-test')
        expect(model, inputs=[x], outputs=[y],
               name='test_greater_or_equal_model')
