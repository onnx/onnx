from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import onnx
from ..base import Base
from . import expect


class GlobalAveragePool(Base):

    @staticmethod
    def export():
        node = onnx.helper.make_node(
            'GlobalAveragePool',
            inputs=['x'],
            outputs=['y'],
        )
        x = np.random.randn(1, 3, 5, 5).astype(np.float32)
        spatial_shape = np.ndim(x) - 2
        y = np.average(x, axis=tuple(range(spatial_shape, spatial_shape + 2)))
        for _ in range(spatial_shape):
            y = np.expand_dims(y, -1)
        expect(node, inputs=[x], outputs=[y], name='test_globalaveragepool')

    @staticmethod
    def export_globalaveragepool_precomputed():

        node = onnx.helper.make_node(
            'GlobalAveragePool',
            inputs=['x'],
            outputs=['y'],
        )
        x = np.array([[[
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]]]).astype(np.float32)
        y = np.array([[[[5]]]]).astype(np.float32)
        expect(node, inputs=[x], outputs=[y], name='test_globalaveragepool_precomputed')
