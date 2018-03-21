from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import onnx
from ..base import Base
from . import expect


class Sigmoid(Base):

    @staticmethod
    def export():
        node = onnx.helper.make_node(
            'Sigmoid',
            inputs=['x'],
            outputs=['y'],
        )

        x = np.array([-1, 0, 1]).astype(np.float32)
        y = 1.0 / (1.0 + np.exp(np.negative(x)))  # expected output [0.26894143, 0.5, 0.7310586]
        expect(node, inputs=[x], outputs=[y],
               name='test_sigmoid_example')

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = 1.0 / (1.0 + np.exp(np.negative(x)))
        expect(node, inputs=[x], outputs=[y],
               name='test_sigmoid')
