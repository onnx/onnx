from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class SingleRelu(Base):

    @staticmethod
    def export():  # type: () -> None

        node = onnx.helper.make_node(
            'Relu', inputs=['x'], outputs=['y'], name='test')
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

    @staticmethod
    def export():  # type: () -> None

        node = onnx.helper.make_node(
            'Relu',
            threshold=1.0,
            value=1.0,
            inputs=['x'],
            outputs=['y'],
            name='test')
        graph = onnx.helper.make_graph(
            nodes=[node],
            name='SingleRelu',
            inputs=[onnx.helper.make_tensor_value_info(
                'x', onnx.TensorProto.FLOAT, [1, 2])],
            outputs=[onnx.helper.make_tensor_value_info(
                'y', onnx.TensorProto.FLOAT, [1, 2])])
        model = onnx.helper.make_model(graph, producer_name='backend-test')

        x = np.random.randn(1, 2).astype(np.float32)
        y = np.clip(x, 1, np.inf).astype(np.float32)

        expect(model, inputs=[x], outputs=[y],
               name='test_single_relu_thresholded_model')
