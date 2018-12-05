from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class Celu(Base):

    @staticmethod
    def export():  # type: () -> None
        node = onnx.helper.make_node(
            'Celu',
            inputs=['x'],
            outputs=['y'],
        )
        x = np.array([-3, 3, -5, 5], dtype=np.float32)
        y = np.array([-0.9502, 3, -0.9933, 5], dtype=np.float32)
        expect(node, inputs=[x], outputs=[y], name='test_celu')
