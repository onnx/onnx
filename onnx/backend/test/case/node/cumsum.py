from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class CumSum(Base):

    @staticmethod
    def export_cumsum_1d():  # type: () -> None
        node = onnx.helper.make_node(
            'CumSum',
            inputs=['x', 'axis'],
            outputs=['y']
        )
        x = np.array([1., 2., 3., 4., 5.]).astype(np.float64)
        axis = np.array([0]).astype(np.int32)
        y = np.array([1., 3., 6., 10., 15.]).astype(np.float64)
        expect(node, inputs=[x, axis], outputs=[y],
               name='test_cumsum_1d')

    @staticmethod
    def export_cumsum_1d_exclusive():  # type: () -> None
        node = onnx.helper.make_node(
            'CumSum',
            inputs=['x', 'axis'],
            outputs=['y'],
            exclusive=1
        )
        x = np.array([1., 2., 3., 4., 5.]).astype(np.float64)
        axis = np.array([0]).astype(np.int32)
        y = np.array([0., 1., 3., 6., 10.]).astype(np.float64)
        expect(node, inputs=[x, axis], outputs=[y],
               name='test_cumsum_1d_exclusive')

    @staticmethod
    def export_cumsum_1d_reverse():  # type: () -> None
        node = onnx.helper.make_node(
            'CumSum',
            inputs=['x', 'axis'],
            outputs=['y'],
            reverse=1
        )
        x = np.array([1., 2., 3., 4., 5.]).astype(np.float64)
        axis = np.array([0]).astype(np.int32)
        y = np.array([15., 14., 12., 9., 5.]).astype(np.float64)
        expect(node, inputs=[x, axis], outputs=[y],
               name='test_cumsum_1d_reverse')

    @staticmethod
    def export_cumsum_1d_reverse_exclusive():  # type: () -> None
        node = onnx.helper.make_node(
            'CumSum',
            inputs=['x', 'axis'],
            outputs=['y'],
            reverse=1
        )
        x = np.array([1., 2., 3., 4., 5.]).astype(np.float64)
        axis = np.array([0]).astype(np.int32)
        y = np.array([14., 12., 9., 5., 0.]).astype(np.float64)
        expect(node, inputs=[x, axis], outputs=[y],
               name='test_cumsum_1d_reverse_exclusive')

    @staticmethod
    def export_cumsum_2d_axis_0():  # type: () -> None
        node = onnx.helper.make_node(
            'CumSum',
            inputs=['x', 'axis'],
            outputs=['y'],
        )
        x = np.array([1., 2., 3., 4., 5., 6.]).astype(np.float64).reshape((2, 3))
        axis = np.array([0]).astype(np.int32)
        y = np.array([1., 2., 3., 5., 7., 9.]).astype(np.float64).reshape((2, 3))
        expect(node, inputs=[x, axis], outputs=[y],
               name='test_cumsum_2d_axis_0')

    @staticmethod
    def export_cumsum_2d_axis_1():  # type: () -> None
        node = onnx.helper.make_node(
            'CumSum',
            inputs=['x', 'axis'],
            outputs=['y'],
        )
        x = np.array([1., 2., 3., 4., 5., 6.]).astype(np.float64).reshape((2, 3))
        axis = np.array([1]).astype(np.int32)
        y = np.array([1., 3., 6., 4., 9., 15.]).astype(np.float64).reshape((2, 3))
        expect(node, inputs=[x, axis], outputs=[y],
               name='test_cumsum_2d_axis_1')

    @staticmethod
    def export_cumsum_2d_negative_axis():  # type: () -> None
        node = onnx.helper.make_node(
            'CumSum',
            inputs=['x', 'axis'],
            outputs=['y'],
        )
        x = np.array([1., 2., 3., 4., 5., 6.]).astype(np.float64).reshape((2, 3))
        axis = np.array([-1]).astype(np.int32)
        y = np.array([1., 3., 6., 4., 9., 15.]).astype(np.float64).reshape((2, 3))
        expect(node, inputs=[x, axis], outputs=[y],
               name='test_cumsum_2d_negative_axis')
