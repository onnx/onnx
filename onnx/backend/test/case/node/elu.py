from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import onnx
from ..base import Base
from . import expect


class Elu(Base):

    @staticmethod
    def export():
        node = onnx.helper.make_node(
            'Elu',
            inputs=['x'],
            outputs=['y'],
            alpha=2.0
        )

        x = np.array([-1, 0, 1]).astype(np.float32)
        # expected output [-1.2642411, 0., 1.]
        y = np.clip(x, 0, np.inf) + (np.exp(np.clip(x, -np.inf, 0)) - 1) * 2.0
        expect(node, inputs=[x], outputs=[y],
               name='test_elu_example')

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.clip(x, 0, np.inf) + (np.exp(np.clip(x, -np.inf, 0)) - 1) * 2.0
        expect(node, inputs=[x], outputs=[y],
               name='test_elu')

    @staticmethod
    def export_elu_default():
        default_alpha = 1.0
        node = onnx.helper.make_node(
            'Elu',
            inputs=['x'],
            outputs=['y'],
        )
        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.clip(x, 0, np.inf) + (np.exp(np.clip(x, -np.inf, 0)) - 1) * default_alpha
        expect(node, inputs=[x], outputs=[y],
               name='test_elu_default')
