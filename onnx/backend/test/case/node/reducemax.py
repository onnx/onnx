from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class ReduceMax(Base):

    @staticmethod
    def export_do_not_keepdims():  # type: () -> None
        shape = [3, 2, 2]
        axes = [1]
        keepdims = 0

        node = onnx.helper.make_node(
            'ReduceMax',
            inputs=['data'],
            outputs=['reduced'],
            axes=axes,
            keepdims=keepdims)

        data = np.array([[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]], dtype=np.float32)
        reduced = np.maximum.reduce(data, axis=tuple(axes), keepdims=keepdims == 1)
        #print(reduced)
        #[[20., 2.]
        # [40., 2.]
        # [60., 2.]]

        expect(node, inputs=[data], outputs=[reduced], name='test_reduce_max_do_not_keepdims_example')

        np.random.seed(0)
        data = np.random.uniform(-10, 10, shape).astype(np.float32)
        reduced = np.maximum.reduce(data, axis=tuple(axes), keepdims=keepdims == 1)

        expect(node, inputs=[data], outputs=[reduced], name='test_reduce_max_do_not_keepdims_random')

    @staticmethod
    def export_keepdims():  # type: () -> None
        shape = [3, 2, 2]
        axes = [1]
        keepdims = 1

        node = onnx.helper.make_node(
            'ReduceMax',
            inputs=['data'],
            outputs=['reduced'],
            axes=axes,
            keepdims=keepdims)

        data = np.array([[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]], dtype=np.float32)
        reduced = np.maximum.reduce(data, axis=tuple(axes), keepdims=keepdims == 1)
        #print(reduced)
        #[[[20., 2.]]
        # [[40., 2.]]
        # [[60., 2.]]]

        expect(node, inputs=[data], outputs=[reduced], name='test_reduce_max_keepdims_example')

        np.random.seed(0)
        data = np.random.uniform(-10, 10, shape).astype(np.float32)
        reduced = np.maximum.reduce(data, axis=tuple(axes), keepdims=keepdims == 1)

        expect(node, inputs=[data], outputs=[reduced], name='test_reduce_max_keepdims_random')

    @staticmethod
    def export_default_axes_keepdims():  # type: () -> None
        shape = [3, 2, 2]
        axes = None
        keepdims = 1
        node = onnx.helper.make_node(
            'ReduceMax',
            inputs=['data'],
            outputs=['reduced'],
            keepdims=keepdims)

        data = np.array([[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]], dtype=np.float32)
        reduced = np.maximum.reduce(data, axis=axes, keepdims=keepdims == 1)
        #print(reduced)
        [[[60.]]]

        expect(node, inputs=[data], outputs=[reduced], name='test_reduce_max_default_axes_keepdim_example')

        np.random.seed(0)
        data = np.random.uniform(-10, 10, shape).astype(np.float32)
        reduced = np.maximum.reduce(data, axis=axes, keepdims=keepdims == 1)

        expect(node, inputs=[data], outputs=[reduced], name='test_reduce_max_default_axes_keepdims_random')
