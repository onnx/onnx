from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore
import onnx
from ..base import Base
from . import expect


class QuantizeLinear(Base):

    @staticmethod
    def export():  # type: () -> None
        node = onnx.helper.make_node('QuantizeLinear',
            inputs=['x', 'y_scale', 'y_zero_point'],
            outputs=['y'],)

        x = np.array([0, 2, 3, 1000, -254, -1000]).astype(np.float32)
        y_scale = np.array([2], dtype=np.float32)
        y_zero_point = np.array([128], dtype=np.uint8)
        y = np.array([128, 129, 130, 255, 1, 0]).astype(np.uint8)

        expect(node, inputs=[x, y_scale, y_zero_point], outputs=[y],
               name='test_quantizelinear')
