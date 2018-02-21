from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import onnx
from ..base import Base
from . import expect


class Softsign(Base):

    @staticmethod
    def export():
        node = onnx.helper.make_node(
            'Softsign',
            inputs=['x'],
            outputs=['y'],
        )

        x = np.array([-1, 0, 1]).astype(np.float32)
        y = np.array([-0.5, 0, 0.5]).astype(np.float32)
        expect(node, inputs=[x], outputs=[y],
               name='test_softsign_example')

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = x / (1 + np.abs(x))
        expect(node, inputs=[x], outputs=[y],
               name='test_softsign')
