from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class EyeLike(Base):

    @staticmethod
    def export_without_dtype():  # type: () -> None
        shape = (4, 4)
        node = onnx.helper.make_node(
            'EyeLike',
            inputs=['x'],
            outputs=['y'],
        )

        x = np.random.randint(0, 100, size=shape, dtype=np.int32)
        y = np.eye(shape[0], shape[1], dtype=np.int32)
        expect(node, inputs=[x], outputs=[y], name='test_eyelike_without_dtype')

    @staticmethod
    def export_with_dtype():  # type: () -> None
        shape = (3, 4)
        node = onnx.helper.make_node(
            'EyeLike',
            inputs=['x'],
            outputs=['y'],
            dtype=onnx.TensorProto.DOUBLE,
        )

        x = np.random.randint(0, 100, size=shape, dtype=np.int32)
        y = np.eye(shape[0], shape[1], dtype=np.float64)
        expect(node, inputs=[x], outputs=[y], name='test_eyelike_with_dtype')

    @staticmethod
    def export_populate_off_main_diagonal():  # type: () -> None
        shape = (4, 5)
        off_diagonal_offset = 1
        node = onnx.helper.make_node(
            'EyeLike',
            inputs=['x'],
            outputs=['y'],
            k=off_diagonal_offset,
            dtype=onnx.TensorProto.FLOAT,
        )

        x = np.random.randint(0, 100, size=shape, dtype=np.int32)
        y = np.eye(shape[0], shape[1], k=off_diagonal_offset, dtype=np.float32)
        expect(node, inputs=[x], outputs=[y], name='test_eyelike_populate_off_main_diagonal')
