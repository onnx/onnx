from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class GatherND(Base):

    @staticmethod
    def export():  # type: () -> None
        node = onnx.helper.make_node(
            'GatherND',
            inputs=['data', 'indices'],
            outputs=['output'],
        )

        data = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=np.float32)
        indices = np.array([[[0, 1]], [[1, 0]]], dtype=np.int64)
        output = np.array([[[2, 3]], [[4, 5]]], dtype=np.float32)
        expect(node, inputs=[data, indices], outputs=[output],
               name='test_gathernd_example')
