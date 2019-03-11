from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class TopK(Base):

    @staticmethod
    def export_top_k():  # type: () -> None
        node = onnx.helper.make_node(
            'TopK',
            inputs=['x', 'k'],
            outputs=['values', 'indices'],
        )
        X = np.array([
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
        ], dtype=np.float32)
        K = np.array([3], dtype=np.int64)
        values_ref = np.array([
            [3, 2, 1],
            [7, 6, 5],
            [11, 10, 9],
        ], dtype=np.float32)
        indices_ref = np.array([
            [3, 2, 1],
            [3, 2, 1],
            [3, 2, 1],
        ], dtype=np.int64)

        expect(node, inputs=[X, K], outputs=[values_ref, indices_ref],
               name='test_top_k')
