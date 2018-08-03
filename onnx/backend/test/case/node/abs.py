from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class Abs(Base):

    @staticmethod
    def export():  # type: () -> None
        node = onnx.helper.make_node(
            'Abs',
            inputs=['x'],
            outputs=['y'],
        )
        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.abs(x)

        expect(node, inputs=[x], outputs=[y],
               name='test_abs')

    @staticmethod
    def export_int8():  # type: () -> None
        node = onnx.helper.make_node(
            'Abs',
            inputs=['x'],
            outputs=['y'],
        )
        x = np.int8([-127,-4,0,3,127])
        y = np.abs(x)

        expect(node, inputs=[x], outputs=[y],
               name='test_abs_int8')