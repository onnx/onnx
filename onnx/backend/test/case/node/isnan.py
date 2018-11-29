from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class IsNaN(Base):

    @staticmethod
    def export():  # type: () -> None
        node = onnx.helper.make_node(
            'IsNaN',
            inputs=['x'],
            outputs=['y'],
        )

        x = np.array([3.0, np.nan, 4.0, np.nan], dtype=np.float32)
        y = np.isnan(x)
        expect(node, inputs=[x], outputs=[y], name='test_isnan')
