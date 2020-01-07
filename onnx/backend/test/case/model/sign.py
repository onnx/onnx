from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect
from typing import Sequence


class SingleSign(Base):

    @staticmethod
    def export():  # type: () -> None
        node = onnx.helper.make_node(
            'Sign', ['x'], ['y'], name='test')

        x = np.array([-1.0, 4.5, -4.5, 3.1, 0.0, 2.4, -5.5]).astype(np.float32)
        y = np.array([-1.0, 1.0, -1.0, 1.0, 0.0, 1.0, -1.0]).astype(np.float32)

        graph = onnx.helper.make_graph(
            nodes=[node],
            name='SingleSign',
            inputs=[onnx.helper.make_tensor_value_info('x',
                                                       onnx.TensorProto.FLOAT,
                                                       [7])],
            outputs=[onnx.helper.make_tensor_value_info('y',
                                                        onnx.TensorProto.FLOAT,
                                                        [7])])
        model = onnx.helper.make_model(graph, producer_name='backend-test')
        expect(model, inputs=[x], outputs=[y],
               name='test_sign_model')
