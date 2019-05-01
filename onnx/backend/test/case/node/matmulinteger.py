from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore
import onnx
from ..base import Base
from . import expect


class MatMulInteger(Base):

    @staticmethod
    def export():  # type: () -> None
        node = onnx.helper.make_node('MatMulInteger',
            inputs=['A', 'B', 'a_zero_point', 'b_zero_point'],
            outputs=['Y'],)

        A = np.array([[11, 7, 3],
            [10, 6, 2],
            [9, 5, 1],
            [8, 4, 0], ], dtype=np.uint8)

        a_zero_point = np.array([12], dtype=np.uint8)

        B = np.array([[1, 4],
            [2, 5],
            [3, 6], ], dtype=np.uint8)

        b_zero_point = np.array([0], dtype=np.uint8)

        output = np.array([[-38, -83],
            [-44, -98],
            [-50, -113],
            [-56, -128], ], dtype=np.int32)

        expect(node, inputs=[A, B, a_zero_point, b_zero_point], outputs=[output],
               name='test_matmulinteger')
