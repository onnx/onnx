from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class Or(Base):

    @staticmethod
    def export():  # type: () -> None
        node = onnx.helper.make_node(
            'Or',
            inputs=['x', 'y'],
            outputs=['or'],
        )

        # 2d
        x = (np.random.randn(3, 4) > 0).astype(np.bool)
        y = (np.random.randn(3, 4) > 0).astype(np.bool)
        z = np.logical_or(x, y)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_or2d')

        # 3d
        x = (np.random.randn(3, 4, 5) > 0).astype(np.bool)
        y = (np.random.randn(3, 4, 5) > 0).astype(np.bool)
        z = np.logical_or(x, y)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_or3d')

        # 4d
        x = (np.random.randn(3, 4, 5, 6) > 0).astype(np.bool)
        y = (np.random.randn(3, 4, 5, 6) > 0).astype(np.bool)
        z = np.logical_or(x, y)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_or4d')

    @staticmethod
    def export_or_broadcast():  # type: () -> None
        node = onnx.helper.make_node(
            'Or',
            inputs=['x', 'y'],
            outputs=['or'],
        )

        # 3d vs 1d
        x = (np.random.randn(3, 4, 5) > 0).astype(np.bool)
        y = (np.random.randn(5) > 0).astype(np.bool)
        z = np.logical_or(x, y)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_or_bcast3v1d')

        # 3d vs 2d
        x = (np.random.randn(3, 4, 5) > 0).astype(np.bool)
        y = (np.random.randn(4, 5) > 0).astype(np.bool)
        z = np.logical_or(x, y)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_or_bcast3v2d')

        # 4d vs 2d
        x = (np.random.randn(3, 4, 5, 6) > 0).astype(np.bool)
        y = (np.random.randn(5, 6) > 0).astype(np.bool)
        z = np.logical_or(x, y)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_or_bcast4v2d')

        # 4d vs 3d
        x = (np.random.randn(3, 4, 5, 6) > 0).astype(np.bool)
        y = (np.random.randn(4, 5, 6) > 0).astype(np.bool)
        z = np.logical_or(x, y)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_or_bcast4v3d')

        # 4d vs 4d
        x = (np.random.randn(1, 4, 1, 6) > 0).astype(np.bool)
        y = (np.random.randn(3, 1, 5, 6) > 0).astype(np.bool)
        z = np.logical_or(x, y)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_or_bcast4v4d')
