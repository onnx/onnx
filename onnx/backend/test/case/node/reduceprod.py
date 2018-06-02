from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class ReduceProd(Base):

    @staticmethod
    def export_do_not_keepdims():  # type: () -> None
        shape = [3, 2, 2]
        axes = [1]
        keepdims = 0

        node = onnx.helper.make_node(
            'ReduceProd',
            inputs=['data'],
            outputs=['reduced'],
            axes=axes,
            keepdims=keepdims)

        data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32)
        reduced = np.prod(data, axis=tuple(axes), keepdims=keepdims == 1)
        #print(reduced)
        #[[3., 8.]
        # [35., 48.]
        # [99., 120.]]

        expect(node, inputs=[data], outputs=[reduced], name='test_reduce_prod_do_not_keepdims_example')

        np.random.seed(0)
        data = np.random.uniform(-10, 10, shape).astype(np.float32)
        reduced = np.prod(data, axis=tuple(axes), keepdims=keepdims == 1)
        expect(node, inputs=[data], outputs=[reduced], name='test_reduce_prod_do_not_keepdims_random')

    @staticmethod
    def export_keepdims():  # type: () -> None
        shape = [3, 2, 2]
        axes = [1]
        keepdims = 1

        node = onnx.helper.make_node(
            'ReduceProd',
            inputs=['data'],
            outputs=['reduced'],
            axes=axes,
            keepdims=keepdims)

        data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32)
        reduced = np.prod(data, axis=tuple(axes), keepdims=keepdims == 1)
        #print(reduced)
        #[[[3., 8.]]
        # [[35., 48.]]
        # [[99., 120.]]]

        expect(node, inputs=[data], outputs=[reduced], name='test_reduce_prod_keepdims_example')

        np.random.seed(0)
        data = np.random.uniform(-10, 10, shape).astype(np.float32)
        reduced = np.prod(data, axis=tuple(axes), keepdims=keepdims == 1)
        expect(node, inputs=[data], outputs=[reduced], name='test_reduce_prod_keepdims_random')

    @staticmethod
    def export_default_axes_keepdims():  # type: () -> None
        shape = [3, 2, 2]
        axes = None
        keepdims = 1

        node = onnx.helper.make_node(
            'ReduceProd',
            inputs=['data'],
            outputs=['reduced'],
            keepdims=keepdims)

        data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32)
        reduced = np.prod(data, axis=axes, keepdims=keepdims == 1)
        #print(reduced)
        #[[[4.790016e+08]]]

        expect(node, inputs=[data], outputs=[reduced], name='test_reduce_prod_default_axes_keepdims_example')

        np.random.seed(0)
        data = np.random.uniform(-10, 10, shape).astype(np.float32)
        reduced = np.prod(data, axis=axes, keepdims=keepdims == 1)
        expect(node, inputs=[data], outputs=[reduced], name='test_reduce_prod_default_axes_keepdims_random')
