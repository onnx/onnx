from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class Upsample(Base):

    @staticmethod
    def export_nearest():  # type: () -> None
        node = onnx.helper.make_node(
            'Upsample',
            inputs=['x'],
            outputs=['y'],
            scales=[1.0, 1.0, 2.0, 3.0],
            mode='nearest',
        )

        data = np.array([[[
            [1, 2],
            [3, 4],
        ]]], dtype=np.float32)

        output = np.array([[[
            [1, 1, 1, 2, 2, 2],
            [1, 1, 1, 2, 2, 2],
            [3, 3, 3, 4, 4, 4],
            [3, 3, 3, 4, 4, 4],
        ]]], dtype=np.float32)

        expect(node, inputs=[data], outputs=[output],
               name='test_upsample_nearest')

    @staticmethod
    def export_linear():  # type: () -> None
        node = onnx.helper.make_node(
            'Upsample',
            inputs=['x'],
            outputs=['y'],
            scales=[1.0, 1.0, 2.0, 2.0],
            mode='linear',
        )

        data = np.array([[[
            [1, 2],
            [3, 4],
        ]]], dtype=np.float32)

        output = np.array([[[
            [1.00, 1.25, 1.75, 2.00],
            [1.50, 1.75, 2.25, 2.50],
            [2.50, 2.75, 3.25, 3.50],
            [3.00, 3.25, 3.75, 4.00],
        ]]], dtype=np.float32)

        expect(node, inputs=[data], outputs=[output],
               name='test_upsample_linear')
