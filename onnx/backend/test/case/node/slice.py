from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import onnx
from .base import Base, expect


class Slice(Base):

    @staticmethod
    def export_slice():
        node = onnx.helper.make_node(
            'Slice',
            inputs=['x'],
            outputs=['y'],
            axes=[0, 1],
            starts=[0, 0],
            ends=[3, 10],
        )

        x = np.random.randn(20, 10, 5).astype(np.float32)
        y = x[0:3, 0:10]

        expect(node, inputs=[x], outputs=[y],
               name='test_slice')

    @staticmethod
    def export_slice_neg():
        node = onnx.helper.make_node(
            'Slice',
            inputs=['x'],
            outputs=['y'],
            axes=[1],
            starts=[0],
            ends=[-1],
        )

        x = np.random.randn(20, 10, 5).astype(np.float32)
        y = x[:, 0:-1]

        expect(node, inputs=[x], outputs=[y],
               name='test_slice_neg')

    @staticmethod
    def export_slice_default_axes():
        node = onnx.helper.make_node(
            'Slice',
            inputs=['x'],
            outputs=['y'],
            starts=[0, 0, 3],
            ends=[20, 10, 4],
        )

        x = np.random.randn(20, 10, 5).astype(np.float32)
        y = x[:, :, 3:4]

        expect(node, inputs=[x], outputs=[y],
               name='test_default_axes')
