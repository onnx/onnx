from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


def arg_max_use_numpy(data, axis=0, keepdims=1):  # type: (np.ndarray, int, int) -> (np.ndarray)
    reduced = np.argmax(data, axis=axis)
    if (keepdims == 1):
        reduced = np.expand_dims(reduced, axis)
    return reduced


class ArgMax(Base):

    @staticmethod
    def export_do_not_keepdims():  # type: () -> None
        shape = [3, 2, 2]
        axis = 1
        keepdims = 0

        node = onnx.helper.make_node(
            'ArgMax',
            inputs=['data'],
            outputs=['reduced'],
            axis=axis,
            keepdims=keepdims)

        data = np.array([[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]], dtype=np.float32)
        reduced = arg_max_use_numpy(data, axis=axis, keepdims=keepdims)
        #print(reduced)
        #[[1 1]
        # [1 1]
        # [1 1]]

        expect(node, inputs=[data], outputs=[reduced], name='test_arg_max_do_not_keepdims_example')

        np.random.seed(0)
        data = np.random.uniform(-10, 10, shape).astype(np.float32)
        reduced = arg_max_use_numpy(data, axis=axis, keepdims=keepdims)

        expect(node, inputs=[data], outputs=[reduced], name='test_arg_max_do_not_keepdims_random')

    @staticmethod
    def export_keepdims():  # type: () -> None
        shape = [3, 2, 2]
        axis = 1
        keepdims = 1

        node = onnx.helper.make_node(
            'ArgMax',
            inputs=['data'],
            outputs=['reduced'],
            axis=axis,
            keepdims=keepdims)

        data = np.array([[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]], dtype=np.float32)
        reduced = arg_max_use_numpy(data, axis=axis, keepdims=keepdims)
        #print(reduced)
        #[[[1 1]]
        # [[1 1]]
        # [[1 1]]]

        expect(node, inputs=[data], outputs=[reduced], name='test_arg_max_keepdims_example')

        np.random.seed(0)
        data = np.random.uniform(-10, 10, shape).astype(np.float32)
        reduced = arg_max_use_numpy(data, axis=axis, keepdims=keepdims)

        expect(node, inputs=[data], outputs=[reduced], name='test_arg_max_keepdims_random')

    @staticmethod
    def export_default_axes_keepdims():  # type: () -> None
        shape = [3, 2, 2]
        keepdims = 1

        node = onnx.helper.make_node(
            'ArgMax',
            inputs=['data'],
            outputs=['reduced'],
            keepdims=keepdims)

        data = np.array([[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]], dtype=np.float32)
        reduced = arg_max_use_numpy(data, keepdims=keepdims)
        #print(reduced)
        #[[[1.]]]

        expect(node, inputs=[data], outputs=[reduced], name='test_arg_max_default_axes_keepdims_example')

        np.random.seed(0)
        data = np.random.uniform(-10, 10, shape).astype(np.float32)
        reduced = arg_max_use_numpy(data, keepdims=keepdims)

        expect(node, inputs=[data], outputs=[reduced], name='test_arg_max_default_axes_keepdims_random')
