from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class Hardmax(Base):

    @staticmethod
    def export():  # type: () -> None
        node = onnx.helper.make_node(
            'Hardmax',
            inputs=['x'],
            outputs=['y'],
        )

        x = np.array([[3, 0, 1, 2], [2, 5, 1, 0], [0, 1, 3, 2], [0, 1, 2, 3]]).astype(np.float32)
        y = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]).astype(np.float32)
        expect(node, inputs=[x], outputs=[y],
               name='test_hardmax_example')

        # For multiple occurrances of the maximal values, the first occurrence is selected for one-hot output
        x = np.array([[3, 3, 3, 1]]).astype(np.float32)
        y = np.array([[1, 0, 0, 0]]).astype(np.float32)
        expect(node, inputs=[x], outputs=[y],
               name='test_hardmax_one_hot')

    @staticmethod
    def export_hardmax_axis():  # type: () -> None
        def hardmax_2d(x):  # type: (np.ndarray) -> np.ndarray
            return np.eye(x.shape[1], dtype=x.dtype)[np.argmax(x, axis=1)]

        x = np.random.randn(3, 4, 5).astype(np.float32)
        node = onnx.helper.make_node(
            'Hardmax',
            inputs=['x'],
            outputs=['y'],
            axis=0,
        )
        y = hardmax_2d(x.reshape(1, 60)).reshape(3, 4, 5)
        expect(node, inputs=[x], outputs=[y],
               name='test_hardmax_axis_0')

        node = onnx.helper.make_node(
            'Hardmax',
            inputs=['x'],
            outputs=['y'],
            axis=1,
        )
        y = hardmax_2d(x.reshape(3, 20)).reshape(3, 4, 5)
        expect(node, inputs=[x], outputs=[y],
               name='test_hardmax_axis_1')

        # default axis is 1
        node = onnx.helper.make_node(
            'Hardmax',
            inputs=['x'],
            outputs=['y'],
        )
        expect(node, inputs=[x], outputs=[y],
               name='test_hardmax_default_axis')

        node = onnx.helper.make_node(
            'Hardmax',
            inputs=['x'],
            outputs=['y'],
            axis=2,
        )
        y = hardmax_2d(x.reshape(12, 5)).reshape(3, 4, 5)
        expect(node, inputs=[x], outputs=[y],
               name='test_hardmax_axis_2')
