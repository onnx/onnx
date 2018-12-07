from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class Greater(Base):

    @staticmethod
    def export():  # type: () -> None
        node = onnx.helper.make_node(
            'Greater',
            inputs=['x', 'y'],
            outputs=['greater'],
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.random.randn(3, 4, 5).astype(np.float32)
        z = np.greater(x, y)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_greater')

    @staticmethod
    def export_greater_broadcast():  # type: () -> None
        node = onnx.helper.make_node(
            'Greater',
            inputs=['x', 'y'],
            outputs=['greater'],
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.random.randn(5).astype(np.float32)
        z = np.greater(x, y)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_greater_bcast')
