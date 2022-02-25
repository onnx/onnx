# SPDX-License-Identifier: Apache-2.0

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class Less(Base):

    @staticmethod
    def export() -> None:
        node = onnx.helper.make_node(
            'Less',
            inputs=['x', 'y'],
            outputs=['less'],
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.random.randn(3, 4, 5).astype(np.float32)
        z = np.less(x, y)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_less')

    @staticmethod
    def export_less_broadcast() -> None:
        node = onnx.helper.make_node(
            'Less',
            inputs=['x', 'y'],
            outputs=['less'],
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.random.randn(5).astype(np.float32)
        z = np.less(x, y)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_less_bcast')
