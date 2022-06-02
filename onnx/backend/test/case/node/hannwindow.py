# SPDX-License-Identifier: Apache-2.0


import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class HannWindow(Base):

    @staticmethod
    def export() -> None:
        node = onnx.helper.make_node(
            'HannWindow',
            inputs=['x'],
            outputs=['y'],
        )
        size = np.int32(10)
        a0 = .5
        a1 = .5
        y = a0 + a1 * np.cos(2 * 3.1415 * np.arange(0, size, 1, dtype=np.float32) / size)
        expect(node, inputs=[size], outputs=[y],
               name='test_hannwindow')
