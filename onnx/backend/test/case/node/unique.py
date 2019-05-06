from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class Unique(Base):

    @staticmethod
    def export():  # type: () -> None
        node = onnx.helper.make_node(
            'Unique',
            inputs=['x'],
            outputs=['y', 'idx'],
        )

        x = np.array([2.0, 1.0, 1.0, 3.0, 4.0, 3.0], dtype=np.float32)
        # numpy unique does not retain original order (it sorts the output unique values)
        # https://github.com/numpy/numpy/issues/8621
        # so going with hand-crafted test case
        y = np.array([2.0, 1.0, 3.0, 4.0], dtype=np.float32)
        idx = np.array([0, 1, 1, 2, 3, 2], dtype=np.int64)
        expect(node, inputs=[x], outputs=[y, idx], name='test_unique_float')
