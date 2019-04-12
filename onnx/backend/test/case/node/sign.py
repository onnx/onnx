from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class Sign(Base):

    @staticmethod
    def export():  # type: () -> None
        node = onnx.helper.make_node(
            'Sign',
            inputs=['x'],
            outputs=['y'],
        )

        x = np.array(range(-5, 6)).astype(np.float32)
        y = np.sign(x)
        expect(node, inputs=[x], outputs=[y],
               name='test_sign')
