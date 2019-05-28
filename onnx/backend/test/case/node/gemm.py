from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class Gemm(Base):

    @staticmethod
    def export_default_no_bias():  # type: () -> None
        node = onnx.helper.make_node(
            'Gemm',
            inputs=['a', 'b', 'c'],
            outputs=['y']
        )
        a = np.random.ranf([3, 5]).astype(np.float32)
        b = np.random.ranf([5, 4]).astype(np.float32)
        c = np.zeros([1,4]).astype(np.float32)
        y = np.dot(a, b) + c
        expect(node, inputs=[a, b, c], outputs=[y],
               name='test_gemm_default_no_bias')

    @staticmethod
    def export_default_scalar_bias():  # type: () -> None
        node = onnx.helper.make_node(
            'Gemm',
            inputs=['a', 'b', 'c'],
            outputs=['y']
        )
        a = np.random.ranf([2, 3]).astype(np.float32)
        b = np.random.ranf([3, 4]).astype(np.float32)
        c = np.random.ranf([1]).astype(np.float32)
        y = np.dot(a, b) + c
        expect(node, inputs=[a, b, c], outputs=[y],
               name='test_gemm_default_scalar_bias')

    @staticmethod
    def export_default_vector_bias():  # type: () -> None
        node = onnx.helper.make_node(
            'Gemm',
            inputs=['a', 'b', 'c'],
            outputs=['y']
        )
        a = np.random.ranf([2, 7]).astype(np.float32)
        b = np.random.ranf([7, 4]).astype(np.float32)
        c = np.random.ranf([1, 4]).astype(np.float32)
        y = np.dot(a, b) + c
        expect(node, inputs=[a, b, c], outputs=[y],
               name='test_gemm_default_vector_bias')

    @staticmethod
    def export_default_matrix_bias():  # type: () -> None
        node = onnx.helper.make_node(
            'Gemm',
            inputs=['a', 'b', 'c'],
            outputs=['y']
        )
        a = np.random.ranf([3, 6]).astype(np.float32)
        b = np.random.ranf([6, 4]).astype(np.float32)
        c = np.random.ranf([3, 4]).astype(np.float32)
        y = np.dot(a, b) + c
        expect(node, inputs=[a, b, c], outputs=[y],
               name='test_gemm_default_matrix_bias')

    @staticmethod
    def export_transposeA(): # type: () -> None
        node = onnx.helper.make_node(
            'Gemm',
            inputs=['a', 'b', 'c'],
            outputs=['y'],
            transA=1
        )
        a = np.random.ranf([6, 3]).astype(np.float32)
        b = np.random.ranf([6, 4]).astype(np.float32)
        c = np.zeros([1,4]).astype(np.float32)
        y = np.dot(a.T, b) + c
        expect(node, inputs=[a, b, c], outputs=[y],
               name='test_gemm_transposeA')

    @staticmethod
    def export_transposeB(): # type: () -> None
        node = onnx.helper.make_node(
            'Gemm',
            inputs=['a', 'b', 'c'],
            outputs=['y'],
            transB=1
        )
        a = np.random.ranf([3, 6]).astype(np.float32)
        b = np.random.ranf([4, 6]).astype(np.float32)
        c = np.zeros([1,4]).astype(np.float32)
        y = np.dot(a, b.T) + c
        expect(node, inputs=[a, b, c], outputs=[y],
               name='test_gemm_transposeB')

    @staticmethod
    def export_alpha():  # type: () -> None
        node = onnx.helper.make_node(
            'Gemm',
            inputs=['a', 'b', 'c'],
            outputs=['y'],
            alpha=0.5
        )
        a = np.random.ranf([3, 5]).astype(np.float32)
        b = np.random.ranf([5, 4]).astype(np.float32)
        c = np.zeros([1,4]).astype(np.float32)
        y = 0.5 * np.dot(a, b) + c
        expect(node, inputs=[a, b, c], outputs=[y],
               name='test_gemm_alpha')

    @staticmethod
    def export_beta():  # type: () -> None
        node = onnx.helper.make_node(
            'Gemm',
            inputs=['a', 'b', 'c'],
            outputs=['y'],
            beta=0.5
        )
        a = np.random.ranf([2, 7]).astype(np.float32)
        b = np.random.ranf([7, 4]).astype(np.float32)
        c = np.random.ranf([1, 4]).astype(np.float32)
        y = np.dot(a, b) + 0.5 * c
        expect(node, inputs=[a, b, c], outputs=[y],
               name='test_gemm_beta')
