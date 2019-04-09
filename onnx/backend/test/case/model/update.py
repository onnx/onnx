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
            'Add', ['x', 'y'], ['z'], name='MyAdd')
        updateNode = onnx.helper.make_node(
            'Update', ['x'], ['z'], name='MyUpdate')

        # Evaluate the graph while Update is ignored.
        x = np.array([4.0, -2.0]).astype(np.float32)
        y = np.array([-1.0, 3.0]).astype(np.float32)
        z = x + y
        # Evaluate Update.
        z = np.copy(x)

        tensorX = onnx.helper.make_tensor_value_info('x', onnx.TensorProto.FLOAT, [2])
        tensorY = onnx.helper.make_tensor_value_info('y', onnx.TensorProto.FLOAT, [2])
        tensorZ = onnx.helper.make_tensor_value_info('z', onnx.TensorProto.FLOAT, [2])
        graph = onnx.helper.make_graph(
            nodes=[node, updateNode],
            name='Update',
            inputs=[tensorX, tensorY],
            outputs=[tensorZ])
        model = onnx.helper.make_model(graph, producer_name='backend-test')
        expect(model, inputs=[x, y], outputs=[z],
               name='test_update')

