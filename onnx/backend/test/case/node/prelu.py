from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class PRelu(Base):

    @staticmethod
    def export():  # type: () -> None
        node = onnx.helper.make_node(
            'PRelu',
            inputs=['x', 'slope'],
            outputs=['y'],
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        slope = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.clip(x, 0, np.inf) + np.clip(x, -np.inf, 0) * slope

        expect(node, inputs=[x, slope], outputs=[y],
               name='test_prelu_example')

    @staticmethod
    def export_prelu_broadcast():  # type: () -> None
        node = onnx.helper.make_node(
            'PRelu',
            inputs=['x', 'slope'],
            outputs=['y'],
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        slope = np.random.randn(5).astype(np.float32)
        y = np.clip(x, 0, np.inf) + np.clip(x, -np.inf, 0) * slope

        expect(node, inputs=[x, slope], outputs=[y],
               name='test_prelu_broadcast')
