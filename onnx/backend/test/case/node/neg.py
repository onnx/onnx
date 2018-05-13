from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class Neg(Base):

    @staticmethod
    def export():  # type: () -> None
        node = onnx.helper.make_node(
            'Neg',
            inputs=['x'],
            outputs=['y'],
        )

        x = np.array([-4, 2]).astype(np.float32)
        y = np.negative(x)  # expected output [4., -2.],
        expect(node, inputs=[x], outputs=[y],
               name='test_neg_example')

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.negative(x)
        expect(node, inputs=[x], outputs=[y],
               name='test_neg')
