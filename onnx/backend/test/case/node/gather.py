from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import onnx
from ..base import Base
from . import expect


class Gather(Base):

    @staticmethod
    def export_gather_0():
        node = onnx.helper.make_node(
            'Gather',
            inputs=['data', 'indices'],
            outputs=['y'],
            axis=0,
        )
        data = np.random.randn(5, 4, 3, 2).astype(np.float32)
        indices = np.array([0, 1, 3])
        y = np.take(data, indices, axis=0)

        expect(node, inputs=[data, indices], outputs=[y],
               name='test_gather_0')

    @staticmethod
    def export_gather_1():
        node = onnx.helper.make_node(
            'Gather',
            inputs=['data', 'indices'],
            outputs=['y'],
            axis=1,
        )
        data = np.random.randn(5, 4, 3, 2).astype(np.float32)
        indices = np.array([0, 1, 3])
        y = np.take(data, indices, axis=1)

        expect(node, inputs=[data, indices], outputs=[y],
               name='test_gather_1')
