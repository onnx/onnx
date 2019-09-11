from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class Unsqueeze(Base):

    @staticmethod
    def export_unsqueeze():  # type: () -> None
        node = onnx.helper.make_node(
            'Unsqueeze',
            inputs=['x'],
            outputs=['y'],
            axes=[0],
        )
        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.expand_dims(x, axis=0)

        expect(node, inputs=[x], outputs=[y],
               name='test_unsqueeze')

    @staticmethod
    def export_unsqueeze_negative_axes():  # type: () -> None
        node = onnx.helper.make_node(
            'Unsqueeze',
            inputs=['x'],
            outputs=['y'],
            axes=[-2],
        )
        x = np.random.randn(1, 3, 1, 5).astype(np.float32)
        y = np.expand_dims(x, axis=-2)
        expect(node, inputs=[x], outputs=[y],
               name='test_unsqueeze_negative_axes')
