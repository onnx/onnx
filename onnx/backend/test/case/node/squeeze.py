from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class Squeeze(Base):

    @staticmethod
    def export_squeeze():  # type: () -> None
        node = onnx.helper.make_node(
            'Squeeze',
            inputs=['x'],
            outputs=['y'],
            axes=[0],
        )
        x = np.random.randn(1, 3, 4, 5).astype(np.float32)
        y = np.squeeze(x, axis=0)

        expect(node, inputs=[x], outputs=[y],
               name='test_squeeze')
