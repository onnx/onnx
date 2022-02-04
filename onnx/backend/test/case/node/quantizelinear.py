# SPDX-License-Identifier: Apache-2.0

import numpy as np  # type: ignore
import onnx
from ..base import Base
from . import expect


class QuantizeLinear(Base):

    @staticmethod
    def export() -> None:
        node = onnx.helper.make_node('QuantizeLinear',
                                     inputs=['x', 'y_scale', 'y_zero_point'],
                                     outputs=['y'],)

        x = np.array([0, 2, 3, 1000, -254, -1000]).astype(np.float32)
        y_scale = np.float32(2)
        y_zero_point = np.uint8(128)
        y = np.array([128, 129, 130, 255, 1, 0]).astype(np.uint8)

        expect(node, inputs=[x, y_scale, y_zero_point], outputs=[y],
               name='test_quantizelinear')

    @staticmethod
    def export_axis() -> None:
        node = onnx.helper.make_node('QuantizeLinear',
                                     inputs=['x', 'y_scale', 'y_zero_point'],
                                     outputs=['y'],)

        x = np.array([[[[-162, 10],
                        [-100, 232],
                        [-20, -50]],

                       [[-76, 0],
                        [0, 252],
                        [32, -44]],

                       [[245, -485],
                        [-960, -270],
                        [-375, -470]], ], ], dtype=np.float32)
        y_scale = np.array([2, 4, 5], dtype=np.float32)
        y_zero_point = np.array([84, 24, 196], dtype=np.uint8)
        y = (x / y_scale.reshape(1, 3, 1, 1) + y_zero_point.reshape(1, 3, 1, 1)).astype(np.uint8)

        expect(node, inputs=[x, y_scale, y_zero_point], outputs=[y],
               name='test_quantizelinear_axis')
