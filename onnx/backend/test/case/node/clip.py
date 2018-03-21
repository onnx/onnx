from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import onnx
from ..base import Base
from . import expect


class Clip(Base):

    @staticmethod
    def export():
        node = onnx.helper.make_node(
            'Clip',
            inputs=['x'],
            outputs=['y'],
            min=-1.0,
            max=1.0
        )

        x = np.array([-2, 0, 2]).astype(np.float32)
        y = np.clip(x, -1, 1)  # expected output [-1., 0., 1.]
        expect(node, inputs=[x], outputs=[y],
               name='test_clip_example')

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.clip(x, -1.0, 1.0)
        expect(node, inputs=[x], outputs=[y],
               name='test_clip')

    @staticmethod
    def export_clip_default():
        node = onnx.helper.make_node(
            'Clip',
            inputs=['x'],
            outputs=['y'],
            min=0.0
        )
        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.clip(x, 0.0, np.inf)
        expect(node, inputs=[x], outputs=[y],
               name='test_clip_default_min')

        node = onnx.helper.make_node(
            'Clip',
            inputs=['x'],
            outputs=['y'],
            max=0.0
        )
        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.clip(x, -np.inf, 0.0)
        expect(node, inputs=[x], outputs=[y],
               name='test_clip_default_max')
