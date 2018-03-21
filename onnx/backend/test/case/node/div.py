from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import onnx
from ..base import Base
from . import expect


class Div(Base):

    @staticmethod
    def export():
        node = onnx.helper.make_node(
            'Div',
            inputs=['x', 'y'],
            outputs=['z'],
        )

        x = np.array([3, 4]).astype(np.float32)
        y = np.array([1, 2]).astype(np.float32)
        z = x / y  # expected output [3., 2.]
        expect(node, inputs=[x, y], outputs=[z],
               name='test_div_example')

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.random.rand(3, 4, 5).astype(np.float32) + 1.0
        z = x / y
        expect(node, inputs=[x, y], outputs=[z],
               name='test_div')

    @staticmethod
    def export_div_broadcast():
        node = onnx.helper.make_node(
            'Div',
            inputs=['x', 'y'],
            outputs=['z'],
            broadcast=1,
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.random.rand(5).astype(np.float32) + 1.0
        z = x / y
        expect(node, inputs=[x, y], outputs=[z],
               name='test_div_bcast')
