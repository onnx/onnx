# SPDX-License-Identifier: Apache-2.0

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class Asin(Base):

    @staticmethod
    def export() -> None:
        node = onnx.helper.make_node(
            'Asin',
            inputs=['x'],
            outputs=['y'],
        )

        x = np.array([-0.5, 0, 0.5]).astype(np.float32)
        y = np.arcsin(x)
        expect(node, inputs=[x], outputs=[y],
               name='test_asin_example')

        x = np.random.rand(3, 4, 5).astype(np.float32)
        y = np.arcsin(x)
        expect(node, inputs=[x], outputs=[y],
               name='test_asin')
