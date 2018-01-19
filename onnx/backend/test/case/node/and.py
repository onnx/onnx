from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import onnx
from .base import Base, expect


class And(Base):

    @staticmethod
    def export():
        node = onnx.helper.make_node(
            'And',
            inputs=['x', 'y'],
            outputs=['and'],
        )

        x = (np.random.randn(3, 4, 5) > 0).astype(np.bool)
        y = (np.random.randn(3, 4, 5) > 0).astype(np.bool)
        z = np.logical_and(x, y)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_and')

    @staticmethod
    def export_and_broadcast():
        node = onnx.helper.make_node(
            'And',
            inputs=['x', 'y'],
            outputs=['and'],
            broadcast=1,
        )

        x = (np.random.randn(3, 4, 5) > 0).astype(np.bool)
        y = (np.random.randn(5) > 0).astype(np.bool)
        z = np.logical_and(x, y)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_and_bcast1d')

        x = (np.random.randn(3, 4, 5) > 0).astype(np.bool)
        y = (np.random.randn(4, 5) > 0).astype(np.bool)
        z = np.logical_and(x, y)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_and_bcast2d')
