from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import onnx
from ..base import Base
from . import expect


class Xor(Base):

    @staticmethod
    def export():
        node = onnx.helper.make_node(
            'Xor',
            inputs=['x', 'y'],
            outputs=['xor'],
        )

        # 2d
        x = (np.random.randn(3, 4) > 0).astype(np.bool)
        y = (np.random.randn(3, 4) > 0).astype(np.bool)
        z = np.logical_xor(x, y)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_xor2d')

        # 3d
        x = (np.random.randn(3, 4, 5) > 0).astype(np.bool)
        y = (np.random.randn(3, 4, 5) > 0).astype(np.bool)
        z = np.logical_xor(x, y)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_xor3d')

        # 4d
        x = (np.random.randn(3, 4, 5, 6) > 0).astype(np.bool)
        y = (np.random.randn(3, 4, 5, 6) > 0).astype(np.bool)
        z = np.logical_xor(x, y)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_xor4d')

    @staticmethod
    def export_xor_broadcast():
        node = onnx.helper.make_node(
            'Xor',
            inputs=['x', 'y'],
            outputs=['xor'],
            broadcast=1,
        )

        # 3d vs 1d
        x = (np.random.randn(3, 4, 5) > 0).astype(np.bool)
        y = (np.random.randn(5) > 0).astype(np.bool)
        z = np.logical_xor(x, y)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_xor_bcast3v1d')

        # 3d vs 2d
        x = (np.random.randn(3, 4, 5) > 0).astype(np.bool)
        y = (np.random.randn(4, 5) > 0).astype(np.bool)
        z = np.logical_xor(x, y)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_xor_bcast3v2d')

        # 4d vs 2d
        x = (np.random.randn(3, 4, 5, 6) > 0).astype(np.bool)
        y = (np.random.randn(5, 6) > 0).astype(np.bool)
        z = np.logical_xor(x, y)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_xor_bcast4v2d')

        # 4d vs 3d
        x = (np.random.randn(3, 4, 5, 6) > 0).astype(np.bool)
        y = (np.random.randn(4, 5, 6) > 0).astype(np.bool)
        z = np.logical_xor(x, y)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_xor_bcast4v3d')

    @staticmethod
    def export_xor_axis():
        x = (np.random.randn(5, 5, 5, 5) > 0).astype(np.bool)
        y = (np.random.randn(5) > 0).astype(np.bool)

        node = onnx.helper.make_node(
            'Xor',
            inputs=['x', 'y'],
            outputs=['xor'],
            broadcast=1,
            axis=0,
        )

        z = np.logical_xor(x, y[:, np.newaxis, np.newaxis, np.newaxis])
        expect(node, inputs=[x, y], outputs=[z],
               name='test_xor_axis0')

        node = onnx.helper.make_node(
            'Xor',
            inputs=['x', 'y'],
            outputs=['xor'],
            broadcast=1,
            axis=1,
        )

        z = np.logical_xor(x, y[:, np.newaxis, np.newaxis, ])
        expect(node, inputs=[x, y], outputs=[z],
               name='test_xor_axis1')

        node = onnx.helper.make_node(
            'Xor',
            inputs=['x', 'y'],
            outputs=['xor'],
            broadcast=1,
            axis=2,
        )

        z = np.logical_xor(x, y[:, np.newaxis, ])
        expect(node, inputs=[x, y], outputs=[z],
               name='test_xor_axis2')

        node = onnx.helper.make_node(
            'Xor',
            inputs=['x', 'y'],
            outputs=['xor'],
            broadcast=1,
            axis=3,
        )

        z = np.logical_xor(x, y)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_xor_axis3')
