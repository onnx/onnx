# SPDX-License-Identifier: Apache-2.0

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


def hardswish(x: np.ndarray) -> np.ndarray:
    alfa = float(1 / 6)
    beta = 0.5
    return x * np.maximum(0, np.minimum(1, alfa * x + beta))


class HardSwish(Base):

    @staticmethod
    def export() -> None:
        node = onnx.helper.make_node(
            'HardSwish',
            inputs=['x'],
            outputs=['y'],
        )
        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = hardswish(x)

        expect(node, inputs=[x], outputs=[y],
               name='test_hardswish')
