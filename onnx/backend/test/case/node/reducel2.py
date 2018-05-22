from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class ReduceL2(Base):

    @staticmethod
    def export_do_not_keepdims():  # type: () -> None
        shape = [3, 2, 2]
        axes = [2]
        keepdims = 0

        node = onnx.helper.make_node(
            'ReduceL2',
            inputs=['data'],
            outputs=['reduced'],
            axes=axes,
            keepdims=keepdims
        )

        data = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape)
        #print(data)
        #[[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]], [[9., 10.], [11., 12.]]]

        reduced = np.sqrt(np.sum(
            a=np.square(data), axis=tuple(axes), keepdims=keepdims == 1))
        #print(reduced)
        #[[2.23606798, 5.],
        # [7.81024968, 10.63014581],
        # [13.45362405, 16.2788206]]

        expect(node, inputs=[data], outputs=[reduced],
            name='test_reduce_l2_do_not_keepdims_example')

        np.random.seed(0)
        data = np.random.uniform(-10, 10, shape).astype(np.float32)
        reduced = np.sqrt(np.sum(
            a=np.square(data), axis=tuple(axes), keepdims=keepdims == 1))

        expect(node, inputs=[data], outputs=[reduced],
            name='test_reduce_l2_do_not_keepdims_random')

    @staticmethod
    def export_keepdims():  # type: () -> None
        shape = [3, 2, 2]
        axes = [2]
        keepdims = 1

        node = onnx.helper.make_node(
            'ReduceL2',
            inputs=['data'],
            outputs=['reduced'],
            axes=axes,
            keepdims=keepdims
        )

        data = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape)
        #print(data)
        #[[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]], [[9., 10.], [11., 12.]]]

        reduced = np.sqrt(np.sum(
            a=np.square(data), axis=tuple(axes), keepdims=keepdims == 1))
        #print(reduced)
        #[[[2.23606798], [5.]]
        # [[7.81024968], [10.63014581]]
        # [[13.45362405], [16.2788206 ]]]

        expect(node, inputs=[data], outputs=[reduced],
            name='test_reduce_l2_keep_dims_example')

        np.random.seed(0)
        data = np.random.uniform(-10, 10, shape).astype(np.float32)
        reduced = np.sqrt(np.sum(
            a=np.square(data), axis=tuple(axes), keepdims=keepdims == 1))

        expect(node, inputs=[data], outputs=[reduced], name='test_reduce_l2_keep_dims_random')

    @staticmethod
    def export_default_axes_keepdims():  # type: () -> None
        shape = [3, 2, 2]
        axes = None
        keepdims = 1

        node = onnx.helper.make_node(
            'ReduceL2',
            inputs=['data'],
            outputs=['reduced'],
            keepdims=keepdims
        )

        data = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape)
        #print(data)
        #[[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]], [[9., 10.], [11., 12.]]]

        reduced = np.sqrt(np.sum(
            a=np.square(data), axis=axes, keepdims=keepdims == 1))
        #print(reduced)
        #[[[25.49509757]]]

        expect(node, inputs=[data], outputs=[reduced],
            name='test_reduce_l2_default_axes_keepdims_example')

        np.random.seed(0)
        data = np.random.uniform(-10, 10, shape).astype(np.float32)
        reduced = np.sqrt(np.sum(
            a=np.square(data), axis=axes, keepdims=keepdims == 1))

        expect(node, inputs=[data], outputs=[reduced],
            name='test_reduce_l2_default_axes_keepdims_random')
