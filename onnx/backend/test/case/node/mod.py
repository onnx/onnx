from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class Mod(Base):

    @staticmethod
    def export():  # type: () -> None
        node = onnx.helper.make_node(
            'Mod',
            inputs=['x', 'y'],
            outputs=['z'],
        )

        x = np.array([4, 7, 5]).astype(np.float32)
        y = np.array([2, 3, 8]).astype(np.float32)
        z = np.mod(x, y)  # expected output [0, 1, 5]
        expect(node, inputs=[x, y], outputs=[z],
               name='test_mod_example')

    @staticmethod
    def export_mul_broadcast():  # type: () -> None
        node = onnx.helper.make_node(
            'Mod',
            inputs=['x', 'y'],
            outputs=['z'],
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.random.randn(1).astype(np.float32)
        z = np.mod(x, y) # expected output [0, 1, 2]
        expect(node, inputs=[x, y], outputs=[z],
               name='test_mod_bcast')
