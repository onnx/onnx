from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class Acos(Base):

    @staticmethod
    def export():  # type: () -> None
        node = onnx.helper.make_node(
            'Acos',
            inputs=['x'],
            outputs=['y'],
        )

        x = np.array([-0.5, 0, 0.5]).astype(np.float32)
        y = np.arccos(x)
        expect(node, inputs=[x], outputs=[y],
               name='test_acos_example')

        x = np.random.rand(3, 4, 5).astype(np.float32)
        y = np.arccos(x)
        expect(node, inputs=[x], outputs=[y],
               name='test_acos')
