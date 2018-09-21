from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class Merge(Base):

    @staticmethod
    def export():  # type: () -> None
        shape = (4, 4, 1)
        node = onnx.helper.make_node(
            'Merge',
            inputs=['x'],
            outputs=['y'],
            axis=1
        )
        new_shape = [-1]
        new_shape.extend(shape[1:])
        x = np.random.randint(0, 100, size=shape, dtype=np.int32)
        y = np.reshape(x, new_shape)
        expect(node, inputs=[x], outputs=[y], name='test_merge')
