from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import onnx
from ..base import Base
from . import expect


class PRelu(Base):

    @staticmethod
    def export():
        node = onnx.helper.make_node(
            'PRelu',
            inputs=['x', 'slope'],
            outputs=['y'],
        )

        x = np.array([[-1, -2], [0, 0], [1, 2]]).astype(np.float32)
        slope = np.array([0.1, 0.2]).astype(np.float32)
        y = np.clip(x, 0, np.inf) + np.clip(x, -np.inf, 0) * slope.reshape(1, 2)

        expect(node, inputs=[x, slope], outputs=[y],
               name='test_prelu_example')

    @staticmethod
    def export_sharing_slope():
        node = onnx.helper.make_node(
            'PRelu',
            inputs=['x', 'slope'],
            outputs=['y'],
        )

        x = np.array([[-1, -2], [0, 0], [1, 2]]).astype(np.float32)
        slope = np.array([0.1]).astype(np.float32)
        y = np.clip(x, 0, np.inf) + np.clip(x, -np.inf, 0) * slope

        expect(node, inputs=[x, slope], outputs=[y],
               name='test_prelu_sharing_slope_example')