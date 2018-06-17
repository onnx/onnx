from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class Sin(Base):

    @staticmethod
    def export():  # type: () -> None
        node = onnx.helper.make_node(
            'Sin',
            inputs=['x'],
            outputs=['y'],
        )

        x = np.array([-1, 0, 1]).astype(np.float32)
        y = np.sin(x)
        expect(node, inputs=[x], outputs=[y],
               name='test_sin_example')

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.sin(x)
        expect(node, inputs=[x], outputs=[y],
               name='test_sin')
