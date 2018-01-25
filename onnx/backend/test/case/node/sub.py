from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import onnx
from ..base import Base
from . import expect

class Sub(Base):

    @staticmethod
    def export():
        node = onnx.helper.make_node(
            'Sub',
            inputs=['x', 'y'],
            outputs=['z'],
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.random.randn(3, 4, 5).astype(np.float32)
        z = x - y
        expect(node, inputs=[x, y], outputs=[z],
               name='test_sub')

    @staticmethod
    def export_sub_broadcast():
        node = onnx.helper.make_node(
            'Sub',
            inputs=['x', 'y'],
            outputs=['z'],
            broadcast=1,
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.random.randn(5).astype(np.float32)
        z = x - y
        expect(node, inputs=[x, y], outputs=[z],
               name='test_sub_bcast')
