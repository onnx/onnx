from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import onnx
from ..base import Base
from . import expect


class Sqrt(Base):

    @staticmethod
    def export():
        node = onnx.helper.make_node(
            'Sqrt',
            inputs=['x'],
            outputs=['y'],
        )

        x = np.array([1, 4, 9]).astype(np.float32)
        y = np.sqrt(x)  # expected output [1., 2., 3.]
        expect(node, inputs=[x], outputs=[y],
               name='test_sqrt_example')

        x = np.abs(np.random.randn(3, 4, 5).astype(np.float32))
        y = np.sqrt(x)
        expect(node, inputs=[x], outputs=[y],
               name='test_sqrt')
