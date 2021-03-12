# SPDX-License-Identifier: Apache-2.0

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect

from onnx.backend.sample.ops.abs import abs


class HardSwish(Base):

    @staticmethod
    def export():  # type: () -> None
        node = onnx.helper.make_node(
            'HardSwish',
            inputs=['x'],
            outputs=['y'],
        )
        x = np.random.randn(3, 4, 5).astype(np.float32)
        alfa= float(1/6)
        beta = 0.5
        y = x*np.maximum(0, np.minimum(1,alfa*x+beta))

        expect(node, inputs=[x], outputs=[y],
               name='test_hardswish')
