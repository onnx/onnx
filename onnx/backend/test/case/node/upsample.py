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
    def export_nearest():
        node = onnx.helper.make_node(
            'Upsample',
            inputs=['x'],
            outputs=['y'],
            height_scale=2.0,
            width_scale=3.0,
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
