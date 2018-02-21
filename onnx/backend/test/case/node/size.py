from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import onnx
from ..base import Base
from . import expect


class Size(Base):

    @staticmethod
    def export():
        node = onnx.helper.make_node(
            'Size',
            inputs=['x'],
            outputs=['y'],
        )

        x = np.array([
            [1, 2, 3],
            [4, 5, 6],
        ]).astype(np.float32)
        y = np.array(6).astype(np.int64)

        expect(node, inputs=[x], outputs=[y],
               name='test_size_example')

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.array(x.size).astype(np.int64)

        expect(node, inputs=[x], outputs=[y],
               name='test_size')
