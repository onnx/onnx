from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore
import random

import onnx
from ..base import Base
from . import expect
from onnx import helper


def dropout(X, drop_probability=0.5, seed=0):  # type: ignore
    if drop_probability == 0:
        return X

    np.random.seed(seed)
    mask = np.random.uniform(0, 1.0, X.shape) >= drop_probability
    scale = (1 / (1 - drop_probability))
    return mask * X * scale


class Dropout(Base):

    @staticmethod
    def export_default():  # type: () -> None
        node = onnx.helper.make_node(
            'Dropout',
            inputs=['x'],
            outputs=['y'],
        )

        x = np.array([-1, 0, 1]).astype(np.float32)
        y = dropout(x)
        expect(node, inputs=[x], outputs=[y],
               name='test_dropout_default')

    @staticmethod
    def export_random():  # type: () -> None
        node = onnx.helper.make_node(
            'Dropout',
            inputs=['x', 'ratio'],
            outputs=['y'],
            seed=0,
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        ratio = np.array(random.uniform(0, 1))
        seed = 0
        y = dropout(x, ratio, seed)

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
               name='test_dropout_default_old', opset_imports=[helper.make_opsetid("", 11)])

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
               name='test_dropout_random_old', opset_imports=[helper.make_opsetid("", 11)])
