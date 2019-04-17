from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore
import onnx
from ..base import Base
from . import expect


class DequantizeLinear(Base):

    @staticmethod
    def export():  # type: () -> None
        node = onnx.helper.make_node('DequantizeLinear',
            inputs=['x', 'x_scale', 'x_zero_point'],
            outputs=['y'],)

        # scalar zero point and scale
        x = np.array([0, 3, 128, 255]).astype(np.uint8)
        x_scale = np.array([2], dtype=np.float32)
        x_zero_point = np.array([128], dtype=np.uint8)
        y = np.array([-256, -250, 0, 254], dtype=np.float32)

        expect(node, inputs=[x, x_scale, x_zero_point], outputs=[y],
               name='test_dequantizelinear')
