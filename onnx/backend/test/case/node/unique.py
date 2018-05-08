from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import onnx
from ..base import Base
from . import expect


class Unique(Base):

    @staticmethod
    def export_unique():
        node = onnx.helper.make_node(
            'Unique',
            inputs=['x'],
            outputs=['y', ''],
        )
        x = np.random.randn(1, 3, 4, 5).astype(np.float32)
        y = np.unique(x)

        expect(node, inputs=[x], outputs=[y],
               name='test_unique')

    @staticmethod
    def export_unique_return_inverse():
        node = onnx.helper.make_node(
            'Unique',
            inputs=['x'],
            outputs=['y', 'inverse'],
            return_inverse=True,
        )
        x = np.random.randn(1, 3, 4, 5).astype(np.float32)
        y, inverse = np.unique(x, return_inverse=True)

        expect(node, inputs=[x], outputs=[y, inverse],
               name='test_unique_return_inverse')
