# SPDX-License-Identifier: Apache-2.0

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class ReduceSum(Base):

    @staticmethod
    def export_do_not_keepdims() -> None:
        shape = [3, 2, 2]
        axes = np.array([1], dtype=np.int64)
        keepdims = 0

        node = onnx.helper.make_node(
            'ReduceSum',
            inputs=['data', 'axes'],
            outputs=['reduced'],
            keepdims=keepdims)

        data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32)
        reduced = np.sum(data, axis=tuple(axes.tolist()), keepdims=keepdims == 1)
        #print(reduced)
        #[[4., 6.]
        # [12., 14.]
        # [20., 22.]]

        expect(node, inputs=[data, axes], outputs=[reduced], name='test_reduce_sum_do_not_keepdims_example')

        np.random.seed(0)
        data = np.random.uniform(-10, 10, shape).astype(np.float32)
        reduced = np.sum(data, axis=tuple(axes.tolist()), keepdims=keepdims == 1)

        expect(node, inputs=[data, axes], outputs=[reduced], name='test_reduce_sum_do_not_keepdims_random')

    @staticmethod
    def export_keepdims() -> None:
        shape = [3, 2, 2]
        axes = np.array([1], dtype=np.int64)
        keepdims = 1

        node = onnx.helper.make_node(
            'ReduceSum',
            inputs=['data', 'axes'],
            outputs=['reduced'],
            keepdims=keepdims)

        data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32)
        reduced = np.sum(data, axis=tuple(axes.tolist()), keepdims=keepdims == 1)
        #print(reduced)
        #[[[4., 6.]]
        # [[12., 14.]]
        # [[20., 22.]]]

        expect(node, inputs=[data, axes], outputs=[reduced], name='test_reduce_sum_keepdims_example')

        np.random.seed(0)
        data = np.random.uniform(-10, 10, shape).astype(np.float32)
        reduced = np.sum(data, axis=tuple(axes.tolist()), keepdims=keepdims == 1)

        expect(node, inputs=[data, axes], outputs=[reduced], name='test_reduce_sum_keepdims_random')

    @staticmethod
    def export_default_axes_keepdims() -> None:
        shape = [3, 2, 2]
        axes = np.array([], dtype=np.int64)
        keepdims = 1

        node = onnx.helper.make_node(
            'ReduceSum',
            inputs=['data', 'axes'],
            outputs=['reduced'],
            keepdims=keepdims)

        data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32)
        reduced = np.sum(data, axis=None, keepdims=keepdims == 1)
        #print(reduced)
        #[[[78.]]]

        expect(node, inputs=[data, axes], outputs=[reduced], name='test_reduce_sum_default_axes_keepdims_example')

        np.random.seed(0)
        data = np.random.uniform(-10, 10, shape).astype(np.float32)
        reduced = np.sum(data, axis=None, keepdims=keepdims == 1)

        expect(node, inputs=[data, axes], outputs=[reduced], name='test_reduce_sum_default_axes_keepdims_random')

    @staticmethod
    def export_negative_axes_keepdims() -> None:
        shape = [3, 2, 2]
        axes = np.array([-2], dtype=np.int64)
        keepdims = 1

        node = onnx.helper.make_node(
            'ReduceSum',
            inputs=['data', 'axes'],
            outputs=['reduced'],
            keepdims=keepdims)

        data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32)
        reduced = np.sum(data, axis=tuple(axes.tolist()), keepdims=keepdims == 1)
        # print(reduced)
        #[[[4., 6.]]
        # [[12., 14.]]
        # [[20., 22.]]]

        expect(node, inputs=[data, axes], outputs=[reduced],
               name='test_reduce_sum_negative_axes_keepdims_example')

        np.random.seed(0)
        data = np.random.uniform(-10, 10, shape).astype(np.float32)
        reduced = np.sum(data, axis=tuple(
            axes.tolist()), keepdims=keepdims == 1)

        expect(node, inputs=[data, axes], outputs=[reduced],
               name='test_reduce_sum_negative_axes_keepdims_random')

    @staticmethod
    def export_empty_axes_input_noop() -> None:
        shape = [3, 2, 2]
        keepdims = 1

        node = onnx.helper.make_node(
            'ReduceSum',
            inputs=['data', 'axes'],
            outputs=['reduced'],
            keepdims=keepdims,
            noop_with_empty_axes=True)

        data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32)
        axes = np.array([], dtype=np.int64)
        reduced = np.array(data)
        #print(reduced)
        #[[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]]

        expect(node, inputs=[data, axes], outputs=[reduced],
               name='test_reduce_sum_empty_axes_input_noop_example')

        np.random.seed(0)
        data = np.random.uniform(-10, 10, shape).astype(np.float32)
        reduced = np.array(data)

        expect(node, inputs=[data, axes], outputs=[reduced], name='test_reduce_sum_negative_axes_keepdims_random')
