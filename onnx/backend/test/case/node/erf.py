from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class Erf(Base):

    @staticmethod
    def export():  # type: () -> None
        node = onnx.helper.make_node(
            'Erf',
            inputs=['x'],
            outputs=['y'],
        )

        x = np.random.randn(1, 3, 32, 32).astype(np.float32)
        y = np.vectorize(math.erf)(x).astype(np.float32)
        expect(node, inputs=[x], outputs=[y],
               name='test_erf')
