from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect
from onnx import helper


class Upsample(Base):

    @staticmethod
    def export_nearest():  # type: () -> None
        node = onnx.helper.make_node(
            'Upsample',
            inputs=['X', 'scales'],
            outputs=['Y'],
            mode='nearest',
        )

        data = np.array([[[
            [1, 2],
            [3, 4],
        ]]], dtype=np.float32)

        scales = np.array([1.0, 1.0, 2.0, 3.0], dtype=np.float32)

        output = np.array([[[
            [1, 1, 1, 2, 2, 2],
            [1, 1, 1, 2, 2, 2],
            [3, 3, 3, 4, 4, 4],
            [3, 3, 3, 4, 4, 4],
        ]]], dtype=np.float32)

        expect(node, inputs=[data, scales], outputs=[output],
               name='test_upsample_nearest', opset_imports=[helper.make_opsetid("", 9)])
