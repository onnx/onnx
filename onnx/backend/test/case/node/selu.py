from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import onnx
from ..base import Base
from . import expect


class Selu(Base):

    @staticmethod
    def export():
        node = onnx.helper.make_node(
            'Selu',
            inputs=['x'],
            outputs=['y'],
            alpha=2.0,
            gamma=3.0
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.clip(x, 0, np.inf) * 3.0 + (np.exp(np.clip(x, -np.inf, 0)) - 1) * 2.0 * 3.0
        expect(node, inputs=[x], outputs=[y],
               name='test_selu')

    @staticmethod
    def export_selu_deault():
        default_alpha = 1.6732
        default_gamma = 1.0507
        node = onnx.helper.make_node(
            'Selu',
            inputs=['x'],
            outputs=['y'],
        )
        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.clip(x, 0, np.inf) * default_gamma + (np.exp(np.clip(x, -np.inf, 0)) - 1) * default_alpha * default_gamma
        expect(node, inputs=[x], outputs=[y],
               name='test_selu_deault')

