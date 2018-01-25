from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import onnx
from ..base import Base
from . import expect


class HardSigmoid(Base):

    @staticmethod
    def export():
        node = onnx.helper.make_node(
            'HardSigmoid',
            inputs=['x'],
            outputs=['y'],
            alpha=0.5,
            beta=0.6
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.clip(x * 0.5 + 0.6, 0, 1)
        expect(node, inputs=[x], outputs=[y],
               name='test_hardsigmoid')

    @staticmethod
    def export_hardsigmoid_deault():
        default_alpha = 0.2
        default_beta=0.5
        node = onnx.helper.make_node(
            'HardSigmoid',
            inputs=['x'],
            outputs=['y'],
        )
        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.clip(x * default_alpha + default_beta, 0, 1)
        expect(node, inputs=[x], outputs=[y],
               name='test_hardsigmoid_deault')

