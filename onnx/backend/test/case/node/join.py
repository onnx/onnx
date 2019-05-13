from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class Join(Base):

    @staticmethod
    def export_join_inner_int():  # type: () -> None
        left = np.array([[1, 10, 11], [2, 20, 21], [3, 30, 31]]).astype(np.int64)
        right = np.array([[4, 94], [2, 92]]).astype(np.int64)
        keys = np.array([0, 0]).astype(np.int64)
        expected = np.array([[2, 20, 21, 92]]).astype(np.int64)

        node = onnx.helper.make_node(
            'Join',
            inputs=['left', 'right', 'keys'],
            outputs=['output']
            # type='INNER'
            # default_int=0
        )
        expect(node, inputs=[left, right, keys], outputs=[expected], name='test_join_inner_int')

    @staticmethod
    def export_join_left_outer_int():  # type: () -> None
        left = np.array([[1, 10, 11], [2, 20, 21], [3, 30, 31]]).astype(np.int64)
        right = np.array([[4, 94], [2, 92]]).astype(np.int64)
        keys = np.array([0, 0]).astype(np.int64)
        expected = np.array([[1, 10, 11, 0], [2, 20, 21, 92], [3, 30, 31, 0]]).astype(np.int64)

        node = onnx.helper.make_node(
            'Join',
            inputs=['left', 'right', 'keys'],
            outputs=['output'],
            type='LEFT_OUTER'
            # default_int=0
        )
        expect(node, inputs=[left, right, keys], outputs=[expected], name='test_join_left_outer_int')

    @staticmethod
    def export_join_right_outer_int():  # type: () -> None
        left = np.array([[1, 10, 11], [2, 20, 21], [3, 30, 31]]).astype(np.int64)
        right = np.array([[4, 94], [2, 92]]).astype(np.int64)
        keys = np.array([0, 0]).astype(np.int64)
        expected = np.array([[4, 0, 0, 94], [2, 20, 21, 92]]).astype(np.int64)

        node = onnx.helper.make_node(
            'Join',
            inputs=['left', 'right', 'keys'],
            outputs=['output'],
            type='RIGHT_OUTER'
            # default_int=0
        )
        expect(node, inputs=[left, right, keys], outputs=[expected], name='test_join_right_outer_int')

    @staticmethod
    def export_join_full_outer_int():  # type: () -> None
        left = np.array([[1, 10, 11], [2, 20, 21], [3, 30, 31]]).astype(np.int64)
        right = np.array([[4, 94], [2, 92]]).astype(np.int64)
        keys = np.array([0, 0]).astype(np.int64)
        expected = np.array([[4, 0, 0, 94], [2, 20, 21, 92]]).astype(np.int64)

        node = onnx.helper.make_node(
            'Join',
            inputs=['left', 'right', 'keys'],
            outputs=['output'],
            type='FULL_OUTER'
            # default_int=0
        )
        expect(node, inputs=[left, right, keys], outputs=[expected], name='test_join_full_outer_int')

    @staticmethod
    def export_join_left_outer_with_default():  # type: () -> None
        left = np.array([[1, 10, 11], [2, 20, 21], [3, 30, 31]]).astype(np.int64)
        right = np.array([[4, 94], [2, 92]]).astype(np.int64)
        keys = np.array([0, 0]).astype(np.int64)
        expected = np.array([[1, 10, 11, -1], [2, 20, 21, 92], [3, 30, 31, -1]]).astype(np.int64)

        node = onnx.helper.make_node(
            'Join',
            inputs=['left', 'right', 'keys'],
            outputs=['output'],
            type='LEFT_OUTER',
            default_int=-1
        )
        expect(node, inputs=[left, right, keys], outputs=[expected], name='test_join_left_outer_with_default')

    @staticmethod
    def export_join_left_outer_with_epsilon():  # type: () -> None
        left = np.array([[1.00, 10.00, 11.00], [2.01, 20.00, 21.00], [3.02, 30.00, 31.00]]).astype(np.float32)
        right = np.array([[4.04, 94.04], [2.02, 92.02]]).astype(np.float32)
        keys = np.array([0, 0]).astype(np.int64)
        expected = np.array([
            [1.00, 10.00, 11.00, 0.00],
            [2.01, 20.00, 21.00, 92.02],
            [3.02, 30.00, 31.00, 0.00]]).astype(np.float32)

        node = onnx.helper.make_node(
            'Join',
            inputs=['left', 'right', 'keys'],
            outputs=['output'],
            type='LEFT_OUTER',
            default_float=0.00,
            epsilon=0.01
        )
        expect(node, inputs=[left, right, keys], outputs=[expected], name='test_join_left_outer_with_default')
