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
    def export_transpose():  # type: () -> None
        node = onnx.helper.make_node(
            'Gemm',
            inputs=['a', 'b', 'c'],
            outputs=['y'],
            alpha=0.5,
            beta=0.5,
            transA=1,
            transB=1
        )
        a = np.random.ranf([6, 3]).astype(np.float32)
        b = np.random.ranf([4, 6]).astype(np.float32)
        c = np.random.ranf([1, 1]).astype(np.float32)
        y = 0.5 * np.dot(a.T, b.T) + 0.5 * c
        expect(node, inputs=[a, b, c], outputs=[y],
               name='test_gemm_broadcast')

    @staticmethod
    def export_notranspose():  # type: () -> None
        node = onnx.helper.make_node(
            'Gemm',
            inputs=['a', 'b', 'c'],
            outputs=['y'],
            alpha=0.5,
            beta=0.5
        )
        a = np.random.ranf([3, 6]).astype(np.float32)
        b = np.random.ranf([6, 4]).astype(np.float32)
        c = np.random.ranf([3, 4]).astype(np.float32)
        y = 0.5 * np.dot(a, b) + 0.5 * c
        expect(node, inputs=[a, b, c], outputs=[y],
               name='test_gemm_nobroadcast')
