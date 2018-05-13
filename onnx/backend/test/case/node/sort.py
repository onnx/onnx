from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import onnx
from ..base import Base
from . import expect


class Sort(Base):

    @staticmethod
    def export_sort():
        node = onnx.helper.make_node(
            'Sort',
            inputs=['x'],
            outputs=['y'],
        )
        x = np.random.randn(1, 3, 4, 5).astype(np.float32)
        y = np.sort(x)

        expect(node, inputs=[x], outputs=[y],
               name='test_sort')

    @staticmethod
    def export_sort_axis():
        node = onnx.helper.make_node(
            'Sort',
            inputs=['x'],
            outputs=['y', 'inverse'],
            axis=2,
        )
        x = np.random.randn(1, 3, 4, 5).astype(np.float32)
        y = np.sort(x, axis=2)

        expect(node, inputs=[x], outputs=[y],
               name='test_sort_axis')
