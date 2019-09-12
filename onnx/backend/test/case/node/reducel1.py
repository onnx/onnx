from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class ReduceL1(Base):

    @staticmethod
    def export_do_not_keepdims():  # type: () -> None
        shape = [3, 2, 2]
        axes = [2]
        keepdims = 0

        node = onnx.helper.make_node(
            'ReduceL1',
            inputs=['data'],
            outputs=['reduced'],
            axes=axes,
            keepdims=keepdims
        )

        data = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape)
        #print(data)
        #[[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]], [[9., 10.], [11., 12.]]]

        reduced = np.sum(a=np.abs(data), axis=tuple(axes), keepdims=keepdims == 1)
        #print(reduced)
        #[[3., 7.], [11., 15.], [19., 23.]]

        expect(node, inputs=[data], outputs=[reduced],
            name='test_reduce_l1_do_not_keepdims_example')

        np.random.seed(0)
        data = np.random.uniform(-10, 10, shape).astype(np.float32)
        reduced = np.sum(a=np.abs(data), axis=tuple(axes), keepdims=keepdims == 1)

        expect(node, inputs=[data], outputs=[reduced],
            name='test_reduce_l1_do_not_keepdims_random')

    @staticmethod
    def export_keepdims():  # type: () -> None
        shape = [3, 2, 2]
        axes = [2]
        keepdims = 1

        node = onnx.helper.make_node(
            'ReduceL1',
            inputs=['data'],
            outputs=['reduced'],
            axes=axes,
            keepdims=keepdims
        )

        data = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape)
        #print(data)
        #[[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]], [[9., 10.], [11., 12.]]]

        reduced = np.sum(a=np.abs(data), axis=tuple(axes), keepdims=keepdims == 1)
        #print(reduced)
        #[[[3.], [7.]], [[11.], [15.]], [[19.], [23.]]]

        expect(node, inputs=[data], outputs=[reduced],
            name='test_reduce_l1_keep_dims_example')

        np.random.seed(0)
        data = np.random.uniform(-10, 10, shape).astype(np.float32)
        reduced = np.sum(a=np.abs(data), axis=tuple(axes), keepdims=keepdims == 1)

        expect(node, inputs=[data], outputs=[reduced],
            name='test_reduce_l1_keep_dims_random')

    @staticmethod
    def export_default_axes_keepdims():  # type: () -> None
        shape = [3, 2, 2]
        axes = None
        keepdims = 1

        node = onnx.helper.make_node(
            'ReduceL1',
            inputs=['data'],
            outputs=['reduced'],
            keepdims=keepdims
        )

        data = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape)
        #print(data)
        #[[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]], [[9., 10.], [11., 12.]]]

        reduced = np.sum(a=np.abs(data), axis=axes, keepdims=keepdims == 1)
        #print(reduced)
        #[[[78.]]]

        expect(node, inputs=[data], outputs=[reduced],
            name='test_reduce_l1_default_axes_keepdims_example')

        np.random.seed(0)
        data = np.random.uniform(-10, 10, shape).astype(np.float32)
        reduced = np.sum(a=np.abs(data), axis=axes, keepdims=keepdims == 1)

        expect(node, inputs=[data], outputs=[reduced],
            name='test_reduce_l1_default_axes_keepdims_random')
