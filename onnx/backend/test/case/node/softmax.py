from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import onnx
from ..base import Base
from . import expect


class Softmax(Base):

    @staticmethod
    def export():
        node = onnx.helper.make_node(
            'Softmax',
            inputs=['x'],
            outputs=['y'],
        )
        x = np.array([[-1, 0, 1]]).astype(np.float32)
        # expected output [[0.09003058, 0.24472848, 0.66524094]]
        y = np.exp(x) / np.sum(np.exp(x), axis=1)
        expect(node, inputs=[x], outputs=[y],
               name='test_softmax_example')

    @staticmethod
    def export_softmax_axis():
        def softmax_2d(x):
            max_x = np.max(x, axis=1).reshape((-1, 1))
            exp_x = np.exp(x - max_x)
            return exp_x / np.sum(exp_x, axis=1).reshape((-1, 1))

        x = np.array([[0, 1, 2, 3], [10000, 10001, 10002, 10003]]).astype(np.float32)
        # expected output [[0.0320586, 0.08714432, 0.23688284, 0.64391428],
        #                 [0.0320586, 0.08714432, 0.23688284, 0.64391428]]
        y = softmax_2d(x)

        node = onnx.helper.make_node(
            'Softmax',
            inputs=['x'],
            outputs=['y'],
        )
        expect(node, inputs=[x], outputs=[y],
               name='test_softmax_large_number')

        x = np.abs(np.random.randn(3, 4, 5).astype(np.float32))
        node = onnx.helper.make_node(
            'Softmax',
            inputs=['x'],
            outputs=['y'],
            axis=0,
        )
        y = softmax_2d(x.reshape(1, 60)).reshape(3, 4, 5)
        expect(node, inputs=[x], outputs=[y],
               name='test_softmax_axis_0')

        node = onnx.helper.make_node(
            'Softmax',
            inputs=['x'],
            outputs=['y'],
            axis=1,
        )
        y = softmax_2d(x.reshape(3, 20)).reshape(3, 4, 5)
        expect(node, inputs=[x], outputs=[y],
               name='test_softmax_axis_1')

        # default axis is 1
        node = onnx.helper.make_node(
            'Softmax',
            inputs=['x'],
            outputs=['y'],
        )
        expect(node, inputs=[x], outputs=[y],
               name='test_softmax_default_axis')

        node = onnx.helper.make_node(
            'Softmax',
            inputs=['x'],
            outputs=['y'],
            axis=2,
        )
        y = softmax_2d(x.reshape(12, 5)).reshape(3, 4, 5)
        expect(node, inputs=[x], outputs=[y],
               name='test_softmax_axis_2')
