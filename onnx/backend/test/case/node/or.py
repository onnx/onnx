from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import onnx
from ..base import Base
from . import expect


class Or(Base):

    @staticmethod
    def export():
        node = onnx.helper.make_node(
            'Or',
            inputs=['x', 'y'],
            outputs=['or'],
        )

        x = (np.random.randn(3, 4, 5) > 0).astype(np.bool)
        y = (np.random.randn(3, 4, 5) > 0).astype(np.bool)
        z = np.logical_or(x, y)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_or')

    @staticmethod
    def export_or_broadcast():
        node = onnx.helper.make_node(
            'Or',
            inputs=['x', 'y'],
            outputs=['or'],
            broadcast=1,
        )

        x = (np.random.randn(3, 4, 5) > 0).astype(np.bool)
        y = (np.random.randn(5) > 0).astype(np.bool)
        z = np.logical_or(x, y)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_or_bcast1d')

        x = (np.random.randn(3, 4, 5) > 0).astype(np.bool)
        y = (np.random.randn(4, 5) > 0).astype(np.bool)
        z = np.logical_or(x, y)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_or_bcast2d')
