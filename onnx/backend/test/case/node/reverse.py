from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class Reverse(Base):

    @staticmethod
    def export_default():  # type: () -> None
        input = np.arange(6.0).reshape(2, 3).astype(np.float32)

        node = onnx.helper.make_node(
            'Reverse',
            inputs=['input'],
            outputs=['output']
        )

        output = np.flip(np.flip(input, 0), 1)

        expect(node, inputs=[input], outputs=[output],
               name='test_reverse_default')

    @staticmethod
    def export_with_axes():  # type: () -> None
        input = np.arange(6.0).reshape(2, 3).astype(np.float32)

        node = onnx.helper.make_node(
            'Reverse',
            inputs=['input'],
            outputs=['output'],
            axes=[0]
        )

        output = np.flip(input, 0)

        expect(node, inputs=[input], outputs=[output],
               name='test_reverse_with_axes')

    @staticmethod
    def export_with_negative_axes():  # type: () -> None
        input = np.arange(12.0).reshape(2, 2, 3).astype(np.float32)

        node = onnx.helper.make_node(
            'Reverse',
            inputs=['input'],
            outputs=['output'],
            axes=[1, -1]
        )

        output = np.flip(np.flip(input, 1), -1)

        expect(node, inputs=[input], outputs=[output],
               name='test_reverse_with_negative_axes')
