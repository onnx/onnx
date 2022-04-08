# SPDX-License-Identifier: Apache-2.0

import numpy as np  # type: ignore
import onnx
from ..base import Base
from . import expect


class DequantizeLinear(Base):

    @staticmethod
    def export() -> None:
        node = onnx.helper.make_node('DequantizeLinear',
                                     inputs=['x', 'x_scale', 'x_zero_point'],
                                     outputs=['y'],)

        # scalar zero point and scale
        x = np.array([0, 3, 128, 255]).astype(np.uint8)
        x_scale = np.float32(2)
        x_zero_point = np.uint8(128)
        y = np.array([-256, -250, 0, 254], dtype=np.float32)

        expect(node, inputs=[x, x_scale, x_zero_point], outputs=[y],
               name='test_dequantizelinear')

    @staticmethod
    def export_axis() -> None:
        node = onnx.helper.make_node('DequantizeLinear',
                                     inputs=['x', 'x_scale', 'x_zero_point'],
                                     outputs=['y'],)

        # 1-D tensor zero point and scale of size equal to axis 1 of the input tensor
        x = np.array([[[[3, 89],
                        [34, 200],
                        [74, 59]],

                       [[5, 24],
                        [24, 87],
                        [32, 13]],

                       [[245, 99],
                        [4, 142],
                        [121, 102]], ], ], dtype=np.uint8)
        x_scale = np.array([2, 4, 5], dtype=np.float32)
        x_zero_point = np.array([84, 24, 196], dtype=np.uint8)
        y = (x.astype(np.float32) - x_zero_point.reshape(1, 3, 1, 1).astype(np.float32)) * x_scale.reshape(1, 3, 1, 1)

        expect(node, inputs=[x, x_scale, x_zero_point], outputs=[y],
               name='test_dequantizelinear_axis')
