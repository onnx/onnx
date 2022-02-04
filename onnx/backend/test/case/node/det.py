# SPDX-License-Identifier: Apache-2.0

import math

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class Det(Base):

    @staticmethod
    def export_2d() -> None:
        node = onnx.helper.make_node(
            'Det',
            inputs=['x'],
            outputs=['y'],
        )

        x = np.arange(4).reshape(2, 2).astype(np.float32)
        y = np.linalg.det(x)  # expect -2
        expect(node, inputs=[x], outputs=[y],
               name='test_det_2d')

    @staticmethod
    def export_nd() -> None:
        node = onnx.helper.make_node(
            'Det',
            inputs=['x'],
            outputs=['y'],
        )

        x = np.array([[[1, 2], [3, 4]], [[1, 2], [2, 1]], [[1, 3], [3, 1]]]).astype(np.float32)
        y = np.linalg.det(x)  # expect array([-2., -3., -8.])
        expect(node, inputs=[x], outputs=[y],
               name='test_det_nd')
