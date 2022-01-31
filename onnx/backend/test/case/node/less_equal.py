# SPDX-License-Identifier: Apache-2.0

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class Less(Base):

    @staticmethod
    def export() -> None:
        node = onnx.helper.make_node(
            'LessOrEqual',
            inputs=['x', 'y'],
            outputs=['less_equal'],
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.random.randn(3, 4, 5).astype(np.float32)
        z = np.less_equal(x, y)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_less_equal')

    @staticmethod
    def export_less_broadcast() -> None:
        node = onnx.helper.make_node(
            'LessOrEqual',
            inputs=['x', 'y'],
            outputs=['less_equal'],
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.random.randn(5).astype(np.float32)
        z = np.less_equal(x, y)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_less_equal_bcast')
