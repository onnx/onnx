from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import onnx
from ..base import Base
from . import expect

class Mul(Base):

    @staticmethod
    def export():
        node = onnx.helper.make_node(
            'Mul',
            inputs=['x', 'y'],
            outputs=['z'],
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.random.randn(3, 4, 5).astype(np.float32)
        z = x * y
        expect(node, inputs=[x, y], outputs=[z],
               name='test_mul')

    @staticmethod
    def export_mul_broadcast():
        node = onnx.helper.make_node(
            'Mul',
            inputs=['x', 'y'],
            outputs=['z'],
            broadcast=1,
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.random.randn(5).astype(np.float32)
        z = x * y
        expect(node, inputs=[x, y], outputs=[z],
               name='test_mul_bcast')
