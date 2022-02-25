# SPDX-License-Identifier: Apache-2.0

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class Ceil(Base):

    @staticmethod
    def export() -> None:
        node = onnx.helper.make_node(
            'Ceil',
            inputs=['x'],
            outputs=['y'],
        )

        x = np.array([-1.5, 1.2]).astype(np.float32)
        y = np.ceil(x)  # expected output [-1., 2.]
        expect(node, inputs=[x], outputs=[y],
               name='test_ceil_example')

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.ceil(x)
        expect(node, inputs=[x], outputs=[y],
               name='test_ceil')
