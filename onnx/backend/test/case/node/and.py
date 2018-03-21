from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import onnx
from ..base import Base
from . import expect


class And(Base):

    @staticmethod
    def export():
        node = onnx.helper.make_node(
            'And',
            inputs=['x', 'y'],
            outputs=['and'],
        )

        # 2d
        x = (np.random.randn(3, 4) > 0).astype(np.bool)
        y = (np.random.randn(3, 4) > 0).astype(np.bool)
        z = np.logical_and(x, y)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_and2d')

        # 3d
        x = (np.random.randn(3, 4, 5) > 0).astype(np.bool)
        y = (np.random.randn(3, 4, 5) > 0).astype(np.bool)
        z = np.logical_and(x, y)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_and3d')

        # 4d
        x = (np.random.randn(3, 4, 5, 6) > 0).astype(np.bool)
        y = (np.random.randn(3, 4, 5, 6) > 0).astype(np.bool)
        z = np.logical_and(x, y)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_and4d')

    @staticmethod
    def export_and_broadcast():
        node = onnx.helper.make_node(
            'And',
            inputs=['x', 'y'],
            outputs=['and'],
            broadcast=1,
        )

        # 3d vs 1d
        x = (np.random.randn(3, 4, 5) > 0).astype(np.bool)
        y = (np.random.randn(5) > 0).astype(np.bool)
        z = np.logical_and(x, y)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_or_bcast3v1d')

        # 3d vs 2d
        x = (np.random.randn(3, 4, 5) > 0).astype(np.bool)
        y = (np.random.randn(4, 5) > 0).astype(np.bool)
        z = np.logical_and(x, y)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_or_bcast3v2d')

        # 4d vs 2d
        x = (np.random.randn(3, 4, 5, 6) > 0).astype(np.bool)
        y = (np.random.randn(5, 6) > 0).astype(np.bool)
        z = np.logical_and(x, y)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_or_bcast4v2d')

        # 4d vs 3d
        x = (np.random.randn(3, 4, 5, 6) > 0).astype(np.bool)
        y = (np.random.randn(4, 5, 6) > 0).astype(np.bool)
        z = np.logical_and(x, y)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_or_bcast4v3d')

    @staticmethod
    def export_and_axis():
        x = (np.random.randn(5, 5, 5, 5) > 0).astype(np.bool)
        y = (np.random.randn(5) > 0).astype(np.bool)

        node = onnx.helper.make_node(
            'And',
            inputs=['x', 'y'],
            outputs=['and'],
            broadcast=1,
            axis=0,
        )

        z = np.logical_and(x, y[:, np.newaxis, np.newaxis, np.newaxis])
        expect(node, inputs=[x, y], outputs=[z],
               name='test_and_axis0')

        node = onnx.helper.make_node(
            'And',
            inputs=['x', 'y'],
            outputs=['and'],
            broadcast=1,
            axis=1,
        )

        z = np.logical_and(x, y[:, np.newaxis, np.newaxis])
        expect(node, inputs=[x, y], outputs=[z],
               name='test_and_axis1')

        node = onnx.helper.make_node(
            'And',
            inputs=['x', 'y'],
            outputs=['and'],
            broadcast=1,
            axis=2,
        )

        z = np.logical_and(x, y[:, np.newaxis])
        expect(node, inputs=[x, y], outputs=[z],
               name='test_and_axis2')

        node = onnx.helper.make_node(
            'And',
            inputs=['x', 'y'],
            outputs=['and'],
            broadcast=1,
            axis=3,
        )

        z = np.logical_and(x, y)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_and_axis3')
