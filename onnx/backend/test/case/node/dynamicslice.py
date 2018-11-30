from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class DynamicSlice(Base):

    @staticmethod
    def export_dynamic_slice():  # type: () -> None
        node = onnx.helper.make_node(
            'DynamicSlice',
            inputs=['x', 'starts', 'ends', 'axes'],
            outputs=['y'],
        )

        x = np.random.randn(20, 10, 5).astype(np.float32)
        y = x[0:3, 0:10]
        starts = np.array([0, 0], dtype=np.int64)
        ends = np.array([3, 10], dtype=np.int64)
        axes = np.array([0, 1], dtype=np.int64)

        expect(node, inputs=[x, starts, ends, axes], outputs=[y],
               name='test_dynamic_slice')

    @staticmethod
    def export_dynamic_slice_neg():  # type: () -> None
        node = onnx.helper.make_node(
            'DynamicSlice',
            inputs=['x', 'starts', 'ends', 'axes'],
            outputs=['y'],
        )

        x = np.random.randn(20, 10, 5).astype(np.float32)
        starts = np.array([0], dtype=np.int64)
        ends = np.array([-1], dtype=np.int64)
        axes = np.array([1], dtype=np.int64)
        y = x[:, 0:-1]

        expect(node, inputs=[x, starts, ends, axes], outputs=[y],
               name='test_dynamic_slice_neg')

    @staticmethod
    def export_dynamic_slice_start_out_of_bounds():  # type: () -> None
        node = onnx.helper.make_node(
            'DynamicSlice',
            inputs=['x', 'starts', 'ends', 'axes'],
            outputs=['y'],
        )

        x = np.random.randn(20, 10, 5).astype(np.float32)
        starts = np.array([1000], dtype=np.int64)
        ends = np.array([1000], dtype=np.int64)
        axes = np.array([1], dtype=np.int64)
        y = x[:, 1000:1000]

        expect(node, inputs=[x, starts, ends, axes], outputs=[y],
               name='test_dynamic_slice_start_out_of_bounds')

    @staticmethod
    def export_dynamic_slice_end_out_of_bounds():  # type: () -> None
        node = onnx.helper.make_node(
            'DynamicSlice',
            inputs=['x', 'starts', 'ends', 'axes'],
            outputs=['y'],
        )

        x = np.random.randn(20, 10, 5).astype(np.float32)
        starts = np.array([1], dtype=np.int64)
        ends = np.array([1000], dtype=np.int64)
        axes = np.array([1], dtype=np.int64)
        y = x[:, 1:1000]

        expect(node, inputs=[x, starts, ends, axes], outputs=[y],
               name='test_dynamic_slice_end_out_of_bounds')

    @staticmethod
    def export_dynamic_slice_default_axes():  # type: () -> None
        node = onnx.helper.make_node(
            'DynamicSlice',
            inputs=['x', 'starts', 'ends'],
            outputs=['y'],
        )

        x = np.random.randn(20, 10, 5).astype(np.float32)
        starts = np.array([0, 0, 3], dtype=np.int64)
        ends = np.array([20, 10, 4], dtype=np.int64)
        y = x[:, :, 3:4]

        expect(node, inputs=[x, starts, ends], outputs=[y],
               name='test_dynamic_slice_default_axes')
