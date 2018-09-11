from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class OnesLike(Base):

    @staticmethod
    def export_without_dtype():  # type: () -> None
        node = onnx.helper.make_node(
            'OnesLike',
            inputs=['x'],
            outputs=['y'],
        )
        shape = (4, 3, 2)
        x = np.random.randint(0, 100, size=shape, dtype=np.int32)
        y = np.ones(shape, dtype=np.int32)
        expect(node, inputs=[x], outputs=[y], name='test_oneslike_dim_without_dtype')

    @staticmethod
    def export_with_dtype():  # type: () -> None
        node = onnx.helper.make_node(
            'OnesLike',
            inputs=['x'],
            outputs=['y'],
            dtype=1, # 1: FLOAT
        )
        shape = (2, 5, 1)
        x = np.random.randint(0, 100, size=shape, dtype=np.int32)
        y = np.ones(shape, dtype=np.float32)
        expect(node, inputs=[x], outputs=[y], name='test_oneslike_dim_with_dtype')