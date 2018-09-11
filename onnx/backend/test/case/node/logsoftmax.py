from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class LogSoftmax(Base):

    @staticmethod
    def export():  # type: () -> None
        node = onnx.helper.make_node(
            'LogSoftmax',
            inputs=['x'],
            outputs=['y'],
        )
        x = np.array([[-1, 0, 1]]).astype(np.float32)
        # expected output [[-2.40760589, -1.40760589, -0.40760589]]
        y = x - np.log(np.sum(np.exp(x), axis=1))
        expect(node, inputs=[x], outputs=[y],
               name='test_logsoftmax_example_1')

    @staticmethod
    def export_logsoftmax_axis():  # type: () -> None
        def logsoftmax_2d(x):  # type: (np.ndarray) -> np.ndarray
            max_x = np.max(x, axis=1).reshape((-1, 1))
            exp_x = np.exp(x - max_x)
            return x - max_x - np.log(np.sum(exp_x, axis=1).reshape((-1, 1)))

        x = np.array([[0, 1, 2, 3], [10000, 10001, 10002, 10003]]).astype(np.float32)
        # expected output [[-3.4401896, -2.4401896, -1.44018972, -0.44018969],
        #                 [-3.4401896, -2.4401896, -1.44018972, -0.44018969]]
        y = logsoftmax_2d(x)

        node = onnx.helper.make_node(
            'LogSoftmax',
            inputs=['x'],
            outputs=['y'],
        )
        expect(node, inputs=[x], outputs=[y],
               name='test_logsoftmax_large_number')

        x = np.abs(np.random.randn(3, 4, 5).astype(np.float32))
        node = onnx.helper.make_node(
            'LogSoftmax',
            inputs=['x'],
            outputs=['y'],
            axis=0,
        )
        y = logsoftmax_2d(x.reshape(1, 60)).reshape(3, 4, 5)
        expect(node, inputs=[x], outputs=[y],
               name='test_logsoftmax_axis_0')

        node = onnx.helper.make_node(
            'LogSoftmax',
            inputs=['x'],
            outputs=['y'],
            axis=1,
        )
        y = logsoftmax_2d(x.reshape(3, 20)).reshape(3, 4, 5)
        expect(node, inputs=[x], outputs=[y],
               name='test_logsoftmax_axis_1')

        # default axis is 1
        node = onnx.helper.make_node(
            'LogSoftmax',
            inputs=['x'],
            outputs=['y'],
        )
        expect(node, inputs=[x], outputs=[y],
               name='test_logsoftmax_default_axis')

        node = onnx.helper.make_node(
            'LogSoftmax',
            inputs=['x'],
            outputs=['y'],
            axis=2,
        )
        y = logsoftmax_2d(x.reshape(12, 5)).reshape(3, 4, 5)
        expect(node, inputs=[x], outputs=[y],
               name='test_logsoftmax_axis_2')
