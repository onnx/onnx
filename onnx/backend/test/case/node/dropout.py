from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect
from onnx import helper


class Dropout(Base):
    @staticmethod
    def export_default():  # type: () -> None
        node = onnx.helper.make_node(
            'Dropout',
            inputs=['x'],
            outputs=['y'],
        )

        x = np.array([-1, 0, 1]).astype(np.float32)
        y = x
        expect(node, inputs=[x], outputs=[y],
               name='test_dropout_default')

    @staticmethod
    def export_random():  # type: () -> None
        node = onnx.helper.make_node(
            'Dropout',
            inputs=['x', 'ratio'],
            outputs=['y'],
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        ratio = np.random.randn(1)
        y = x
        expect(node, inputs=[x, ratio], outputs=[y],
               name='test_dropout_random')

    @staticmethod
    def export_default_old():  # type: () -> None
        node = onnx.helper.make_node(
            'Dropout',
            inputs=['x'],
            outputs=['y'],
        )

        x = np.array([-1, 0, 1]).astype(np.float32)
        y = x
        expect(node, inputs=[x], outputs=[y],
               name='test_dropout_default_old',  opset_imports=[helper.make_opsetid("", 10)])

    @staticmethod
    def export_random_old():  # type: () -> None
        node = onnx.helper.make_node(
            'Dropout',
            inputs=['x'],
            outputs=['y'],
            ratio=.2,
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = x
        expect(node, inputs=[x], outputs=[y],
               name='test_dropout_random_old',  opset_imports=[helper.make_opsetid("", 10)])
