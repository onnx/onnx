from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class Shrink(Base):

    @staticmethod
    def export_hard_shrink():  # type: () -> None
        node = onnx.helper.make_node(
            'Shrink',
            inputs=['x'],
            outputs=['y'],
            lambd=1.5,
        )
        X = np.arange(-2.0, 2.1, dtype=np.float32)
        Y = np.array([-2, 0, 0, 0, 2], dtype=np.float32)
        expect(node, inputs=[X], outputs=[Y],
               name='test_shrink_hard')

    @staticmethod
    def export_soft_shrink():  # type: () -> None
        node = onnx.helper.make_node(
            'Shrink',
            inputs=['x'],
            outputs=['y'],
            lambd=1.5,
            bias=1.5,
        )
        X = np.arange(-2.0, 2.1, dtype=np.float32)
        Y = np.array([-0.5, 0, 0, 0, 0.5], dtype=np.float32)
        expect(node, inputs=[X], outputs=[Y],
               name='test_shrink_soft')
