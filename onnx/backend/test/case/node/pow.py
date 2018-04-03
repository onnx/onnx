from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import onnx
from ..base import Base
from . import expect


class Pow(Base):

    @staticmethod
    def export():
        node = onnx.helper.make_node(
            'Pow',
            inputs=['x', 'y'],
            outputs=['z'],
        )

        x = np.array([1, 2, 3]).astype(np.float32)
        y = np.array([4, 5, 6]).astype(np.float32)
        z = np.power(x, y)  # expected output [1., 32., 729.]
        expect(node, inputs=[x, y], outputs=[z],
               name='test_pow_example')

        x = np.arange(60).reshape(3, 4, 5).astype(np.float32)
        y = np.random.randn(3, 4, 5).astype(np.float32)
        z = np.power(x, y)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_pow')

    @staticmethod
    def export_pow_broadcast():
        node = onnx.helper.make_node(
            'Pow',
            inputs=['x', 'y'],
            outputs=['z'],
            broadcast=1,
        )

        x = np.array([1, 2, 3]).astype(np.float32)
        y = np.array([2]).astype(np.float32)
        z = np.power(x, y)  # expected output [1., 4., 9.]
        expect(node, inputs=[x, y], outputs=[z],
               name='test_pow_bcast')

        node = onnx.helper.make_node(
            'Pow',
            inputs=['x', 'y'],
            outputs=['z'],
            broadcast=1,
            axis=0,
        )
        x = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32)
        y = np.array([2, 3]).astype(np.float32)
        z = np.array([[1, 4, 9], [64, 125, 216]]).astype(np.float32)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_pow_bcast_axis0')
