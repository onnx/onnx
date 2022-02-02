# SPDX-License-Identifier: Apache-2.0

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class Add(Base):

    @staticmethod
    def export() -> None:
        node = onnx.helper.make_node(
            'Add',
            inputs=['x', 'y'],
            outputs=['sum'],
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.random.randn(3, 4, 5).astype(np.float32)
        expect(node, inputs=[x, y], outputs=[x + y],
               name='test_add')

    @staticmethod
    def export_add_uint8() -> None:
        node = onnx.helper.make_node(
            'Add',
            inputs=['x', 'y'],
            outputs=['sum'],
        )

        x = np.random.randint(24, size=(3, 4, 5), dtype=np.uint8)
        y = np.random.randint(24, size=(3, 4, 5), dtype=np.uint8)
        expect(node, inputs=[x, y], outputs=[x + y],
               name='test_add_uint8')

    @staticmethod
    def export_add_broadcast() -> None:
        node = onnx.helper.make_node(
            'Add',
            inputs=['x', 'y'],
            outputs=['sum'],
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = np.random.randn(5).astype(np.float32)
        expect(node, inputs=[x, y], outputs=[x + y],
               name='test_add_bcast')
