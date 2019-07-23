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
        Y_Scale = np.array([0.0196078438]).astype(np.float32)
        Y_ZeroPoint = np.array([153]).astype(np.uint8)

        expect(node, inputs=[X], outputs=[Y, Y_Scale, Y_ZeroPoint],
               name='test_dynamicquantizelinear')

        X = np.array([0, 2.5, 3.2, 4.6, 4.8, 5]).astype(np.float32)
        Y = np.array([0, 127, 163, 235, 245, 255]).astype(np.uint8)
        Y_Scale = np.array([0.0196]).astype(np.float32)
        Y_ZeroPoint = np.array([0]).astype(np.uint8)

        expect(node, inputs=[X], outputs=[Y, Y_Scale, Y_ZeroPoint],
               name='test_dynamicquantizelinear_zeropoint_0')

        X = np.array([1, 2.5, 3.2, 4.6, 4.8, 5]).astype(np.float32)
        Y = np.array([51, 127, 163, 235, 245, 255]).astype(np.uint8)
        Y_Scale = np.array([0.0196]).astype(np.float32)
        Y_ZeroPoint = np.array([0]).astype(np.uint8)

        expect(node, inputs=[X], outputs=[Y, Y_Scale, Y_ZeroPoint],
               name='test_dynamicquantizelinear_min_adjusted')
