from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import numpy as np  # type: ignore
import onnx
from ..base import Base
from . import expect


class LRN(Base):

    @staticmethod
    def export_l2normalization_axis_0():  # type: () -> None
        node = onnx.helper.make_node(
            'LpNormalization',
            inputs=['x'],
            outputs=['y'],
            epsilon = 1e-5,
            axis = 0,
            p = 2
        )
        x = np.random.randn(3, 4, 5)
        l2_norm_axis_0 = np.sqrt(np.sum(x**2, axis=0, keepdims=True)) + 1e-5
        y = x / l2_norm_axis_0
        expect(node, inputs=[x], outputs=[y],
               name='test_l2normalization_axis_0')

    @staticmethod
    def export_l2normalization_axis_1():  # type: () -> None
        node = onnx.helper.make_node(
            'LpNormalization',
            inputs=['x'],
            outputs=['y'],
            epsilon = 1e-5,
            axis = 1,
            p = 2
        )
        x = np.random.randn(3, 4, 5)
        l2_norm_axis_1 = np.sqrt(np.sum(x**2, axis=1, keepdims=True)) + 1e-5
        y = x / l2_norm_axis_1
        expect(node, inputs=[x], outputs=[y],
               name='test_l2normalization_axis_1')

    @staticmethod
    def export_l2normalization_axis_last():  # type: () -> None
        node = onnx.helper.make_node(
            'LpNormalization',
            inputs=['x'],
            outputs=['y'],
            epsilon = 1e-5,
            axis = -1,
            p = 2
        )
        x = np.random.randn(3, 4, 5)
        l2_norm_axis_last = np.sqrt(np.sum(x**2, axis=-1, keepdims=True)) + 1e-5
        y = x / l2_norm_axis_last
        expect(node, inputs=[x], outputs=[y],
               name='test_l2normalization_axis_last')

    @staticmethod
    def export_l1normalization_axis_0():  # type: () -> None
        node = onnx.helper.make_node(
            'LpNormalization',
            inputs=['x'],
            outputs=['y'],
            epsilon = 1e-5,
            axis = 0,
            p = 1
        )
        x = np.random.rand(3, 4, 5)
        l1_norm_axis_0 = np.sum(abs(x), axis=0, keepdims=True) + 1e-5
        y = x / l1_norm_axis_0
        expect(node, inputs=[x], outputs=[y],
               name='test_l1normalization_axis_0')

    @staticmethod
    def export_l1normalization_axis_1():  # type: () -> None
        node = onnx.helper.make_node(
            'LpNormalization',
            inputs=['x'],
            outputs=['y'],
            epsilon = 1e-5,
            axis = 1,
            p = 1
        )
        x = np.random.rand(3, 4, 5)
        l1_norm_axis_1 = np.sum(abs(x), axis=1, keepdims=True) + 1e-5
        y = x / l1_norm_axis_1
        expect(node, inputs=[x], outputs=[y],
               name='test_l1normalization_axis_1')

    @staticmethod
    def export_l1normalization_axis_last():  # type: () -> None
        node = onnx.helper.make_node(
            'LpNormalization',
            inputs=['x'],
            outputs=['y'],
            epsilon = 1e-5,
            axis = -1,
            p = 1
        )
        x = np.random.rand(3, 4, 5)
        l1_norm_axis_last = np.sum(abs(x), axis=-1, keepdims=True) + 1e-5
        y = x / l1_norm_axis_last
        expect(node, inputs=[x], outputs=[y],
               name='test_l2normalization_axis_last')

    @staticmethod
    def export_default():  # type: () -> None
        x = np.random.rand(3, 4, 5)

        node = onnx.helper.make_node(
            'LpNormalization',
            inputs=['x'],
            outputs=['y']
        )
        lp_norm_default = np.sqrt(np.sum(x**2, axis=-1, keepdims=True)) + 1e-10
        y = x / lp_norm_default
        expect(node, inputs=[x], outputs=[y],
               name='test_lpnormalization_default')
