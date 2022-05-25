# SPDX-License-Identifier: Apache-2.0


import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class BlackmanWindow(Base):

    @staticmethod
    def export() -> None:
        node = onnx.helper.make_node(
            'BlackmanWindow',
            inputs=['x'],
            outputs=['y'],
        )
        size = np.int32(10)
        a0 = 7938 / 18608
        a1 = 9240 / 18608
        a2 = 1430 / 18608
        y = a0
        y += a1 * np.cos(2 * 3.1415 * np.arange(0, size, 1, dtype=np.float32) / size)
        y += a2 * np.cos(4 * 3.1415 * np.arange(0, size, 1, dtype=np.float32) / size)
        expect(node, inputs=[size], outputs=[y],
               name='test_blackmanwindow')
