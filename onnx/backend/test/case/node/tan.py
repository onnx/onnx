# SPDX-License-Identifier: Apache-2.0

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class Tan(Base):

    @staticmethod
    def export() -> None:
        node = onnx.helper.make_node(
            'Tan',
            inputs=['x'],
            outputs=['y'],
        )

        x = np.array([-1, 0, 1]).astype(np.float32)
        y = np.tan(x)
        expect(node, inputs=[x], outputs=[y],
               name='test_tan_example')

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.tan(x)
        expect(node, inputs=[x], outputs=[y],
               name='test_tan')
