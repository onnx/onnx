from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


def argmin_use_numpy(data, axis=0, keepdims=1):  # type: (np.ndarray, int, int) -> (np.ndarray)
    result = np.argmin(data, axis=axis)
    if (keepdims == 1):
        result = np.expand_dims(result, axis)
    return result.astype(np.int64)


class ArgMin(Base):

    @staticmethod
    def export_no_keepdims():  # type: () -> None
        data = np.array([[2, 1], [3, 10]], dtype=np.float32)
        axis = 1
        keepdims = 0
        node = onnx.helper.make_node(
            'ArgMin',
            inputs=['data'],
            outputs=['result'],
            axis=axis,
            keepdims=keepdims)
        # result: [[1, 0]]
        result = argmin_use_numpy(data, axis=axis, keepdims=keepdims)
        expect(node, inputs=[data], outputs=[result], name='test_argmin_no_keepdims_example')

        data = np.random.uniform(-10, 10, [2, 3, 4]).astype(np.float32)
        # result's shape: [2, 4]
        result = argmin_use_numpy(data, axis=axis, keepdims=keepdims)
        expect(node, inputs=[data], outputs=[result], name='test_argmin_no_keepdims_random')

    @staticmethod
    def export_keepdims():  # type: () -> None
        data = np.array([[2, 1], [3, 10]], dtype=np.float32)
        axis = 1
        keepdims = 1
        node = onnx.helper.make_node(
            'ArgMin',
            inputs=['data'],
            outputs=['result'],
            axis=axis,
            keepdims=keepdims)
        # result: [[1], [0]]
        result = argmin_use_numpy(data, axis=axis, keepdims=keepdims)
        expect(node, inputs=[data], outputs=[result], name='test_argmin_keepdims_example')

        data = np.random.uniform(-10, 10, [2, 3, 4]).astype(np.float32)
        # result's shape: [2, 1, 4]
        result = argmin_use_numpy(data, axis=axis, keepdims=keepdims)
        expect(node, inputs=[data], outputs=[result], name='test_argmin_keepdims_random')

    @staticmethod
    def export_default_axes_keepdims():  # type: () -> None
        data = np.array([[2, 1], [3, 10]], dtype=np.float32)
        keepdims = 1
        node = onnx.helper.make_node(
            'ArgMin',
            inputs=['data'],
            outputs=['result'],
            keepdims=keepdims)

        # result: [[0], [0]]
        result = argmin_use_numpy(data, keepdims=keepdims)
        expect(node, inputs=[data], outputs=[result], name='test_argmin_default_axis_example')

        data = np.random.uniform(-10, 10, [2, 3, 4]).astype(np.float32)
        # result's shape: [1, 3, 4]
        result = argmin_use_numpy(data, keepdims=keepdims)
        expect(node, inputs=[data], outputs=[result], name='test_argmin_default_axis_random')
