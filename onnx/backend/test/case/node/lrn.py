from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class LRN(Base):

    @staticmethod
    def export():
        node = onnx.helper.make_node(
            'LRN',
            inputs=['x'],
            outputs=['y'],
            alpha=0.0002,
            beta=0.5,
            bias=2.0,
            size=1
        )
        x = np.random.ranf([5, 5, 5, 5]).astype(np.float32) * 1000
        y = [i/((2 + 0.0002 * i ** 2) ** 0.5) for i in x][0]
        expect(node, inputs=[x], outputs=[y],
               name='test_lrn')

    @staticmethod
    def export_default():
        node = onnx.helper.make_node(
            'LRN',
            inputs=['x'],
            outputs=['y'],
            alpha=0.0002,
            beta=0.5,
            size=1
        )
        x = np.random.ranf([5, 5, 5, 5]).astype(np.float32) * 1000
        y = [i/((1 + 0.0002 * i ** 2) ** 0.5) for i in x][0]
        expect(node, inputs=[x], outputs=[y],
               name='test_lrn_default')
