from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class Mod(Base):

    @staticmethod
    def export_float_mixed_sign():  # type: () -> None
        node = onnx.helper.make_node(
            'Mod',
            inputs=['x', 'y'],
            outputs=['z'],
        )

        x = np.array([-4.3, 7.2, 5.0, 4.3, -7.2, 8.0])
        y = np.array([2.1, -3.4, 8.0, -2.1, 3.4, 5.0])
        z = np.mod(x, y)  # expected output [2., -3.,  5., -2.,  3.,  3.]
        expect(node, inputs=[x, y], outputs=[z],
               name='test_mod_float_mixed_sign_example')

    @staticmethod
    def export_fmod_mixed_sign():  # type: () -> None
        node = onnx.helper.make_node(
            'Mod',
            inputs=['x', 'y'],
            outputs=['z'],
            fmod=1
        )

        x = np.array([-4.3, 7.2, 5.0, 4.3, -7.2, 8.0])
        y = np.array([2.1, -3.4, 8.0, -2.1, 3.4, 5.0])
        z = np.fmod(x, y)  # expected output [-0.1,  0.4,  5. ,  0.1, -0.4,  3.]
        expect(node, inputs=[x, y], outputs=[z],
               name='test_mod_fmod_mixed_sign_example')

    @staticmethod
    def export_int64_mixed_sign():  # type: () -> None
        node = onnx.helper.make_node(
            'Mod',
            inputs=['x', 'y'],
            outputs=['z'],
        )

        x = np.array([-4, 7, 5, 4, -7, 8]).astype(np.int64)
        y = np.array([2, -3, 8, -2, 3, 5]).astype(np.int64)
        z = np.mod(x, y)  # expected output [ 0, -2,  5,  0,  2,  3]
        expect(node, inputs=[x, y], outputs=[z],
               name='test_mod_int64_mixed_sign_example')

    @staticmethod
    def export_mul_broadcast():  # type: () -> None
        node = onnx.helper.make_node(
            'Mod',
            inputs=['x', 'y'],
            outputs=['z'],
        )

        x = np.arange(0, 30).reshape([3, 2, 5])
        y = np.array([7])
        z = np.mod(x, y)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_mod_bcast')
