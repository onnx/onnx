# SPDX-License-Identifier: Apache-2.0

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class Identity(Base):

    @staticmethod
    def export():  # type: () -> None
        node = onnx.helper.make_node(
            'Identity',
            inputs=['x'],
            outputs=['y'],
        )

        data = np.array([[[
            [1, 2],
            [3, 4],
        ]]], dtype=np.float32)

        expect(node, inputs=[data], outputs=[data],
               name='test_identity')

    @staticmethod
    def export_sequence():  # type: () -> None
        node = onnx.helper.make_node(
            'Identity',
            inputs=['x'],
            outputs=['y'],
        )

        data = [
            np.array([[[
                [1, 2],
                [3, 4],
            ]]], dtype=np.float32),
            np.array([[[
                [2, 3],
                [1, 5],
            ]]], dtype=np.float32)]

        expect(node, inputs=[data], outputs=[data], name='test_identity_sequence')
