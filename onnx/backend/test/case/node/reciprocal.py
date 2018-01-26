from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import onnx
from ..base import Base
from . import expect


class Reciprocal(Base):

    @staticmethod
    def export():
        node = onnx.helper.make_node(
            'Reciprocal',
            inputs=['x'],
            outputs=['y'],
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        x[x==0] = 1
        y = np.reciprocal(x)
        expect(node, inputs=[x], outputs=[y],
               name='test_reciprocal')
