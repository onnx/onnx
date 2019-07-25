from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from onnx import TensorProto
from ..base import Base
from . import expect


class DynamicQuantizeLinear(Base):

    @staticmethod
    def export():  # type: () -> None
        node = onnx.helper.make_node('DynamicQuantizeLinear',
            inputs=['x'],
            outputs=['y', 'y_scale', 'y_zero_point'],
            to=np.int64(2))

        X = np.array([0, 2, -3, -2.5, 1.34, 0.5]).astype(np.float32)
        Y = np.array([153, 255, 0, 26, 221, 179]).astype(np.uint8)
        Y_Scale = np.float32(0.0196078438)
        Y_ZeroPoint = np.uint8(153)

        expect(node, inputs=[X], outputs=[Y, Y_Scale, Y_ZeroPoint],
               name='test_dynamicquantizelinear')

        X = np.array([-1.0, -2.1, -1.3, -2.5, -3.34, -4.0]).astype(np.float32)
        Y = np.array([191, 121, 172, 96, 42, 0]).astype(np.uint8)
        Y_Scale = np.float32(0.0156862754)
        Y_ZeroPoint = np.uint8(255)

        expect(node, inputs=[X], outputs=[Y, Y_Scale, Y_ZeroPoint],
               name='test_dynamicquantizelinear_max_adjusted')

        X = np.array([1, 2.1, 1.3, 2.5,
                      3.34, 4.0, 1.5, 2.6,
                      3.9, 4.0, 3.0, 2.345]).astype(np.float32).reshape((3, 4))

        Y = np.array([64, 134, 83, 159,
                      213, 255, 96, 166,
                      249, 255, 191, 149]).astype(np.uint8).reshape((3, 4))

        Y_Scale = np.float32(0.0156862754)
        Y_ZeroPoint = np.uint8(0)

        expect(node, inputs=[X], outputs=[Y, Y_Scale, Y_ZeroPoint],
               name='test_dynamicquantizelinear_min_adjusted')
