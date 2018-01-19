from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import onnx
from .base import Base, expect


class Xor(Base):

    @staticmethod
    def export():
        node = onnx.helper.make_node(
            'Xor',
            inputs=['x', 'y'],
            outputs=['xor'],
        )

        x = (np.random.randn(3, 4, 5) > 0).astype(np.bool)
        y = (np.random.randn(3, 4, 5) > 0).astype(np.bool)
        z = np.logical_xor(x, y)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_xor')

    @staticmethod
    def export_and_broadcast():
        node = onnx.helper.make_node(
            'Xor',
            inputs=['x', 'y'],
            outputs=['xor'],
            broadcast=1,
        )

        x = (np.random.randn(3, 4, 5) > 0).astype(np.bool)
        y = (np.random.randn(5) > 0).astype(np.bool)
        z = np.logical_xor(x, y)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_xor_bcast1d')

        x = (np.random.randn(3, 4, 5) > 0).astype(np.bool)
        y = (np.random.randn(4, 5) > 0).astype(np.bool)
        z = np.logical_xor(x, y)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_xor_bcast2d')
