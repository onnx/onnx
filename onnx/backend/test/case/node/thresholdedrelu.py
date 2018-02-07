from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import onnx
from ..base import Base
from . import expect


class ThresholdedRelu(Base):

    @staticmethod
    def export():
        alpha = 2.0
        node = onnx.helper.make_node(
            'ThresholdedRelu',
            inputs=['x'],
            outputs=['y'],
            alpha=alpha
        )
        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.clip(x, alpha, np.inf)

        expect(node, inputs=[x], outputs=[y],
               name='test_thresholdedrelu')

    @staticmethod
    def export_default():
        default_alpha = 1.0
        node = onnx.helper.make_node(
            'ThresholdedRelu',
            inputs=['x'],
            outputs=['y']
        )
        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.clip(x, default_alpha, np.inf)

        expect(node, inputs=[x], outputs=[y],
               name='test_thresholdedrelu_default')
