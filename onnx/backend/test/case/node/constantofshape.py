# SPDX-License-Identifier: Apache-2.0

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class ConstantOfShape(Base):

    @staticmethod
    def export_float_ones() -> None:
        x = np.array([4, 3, 2]).astype(np.int64)
        tensor_value = onnx.helper.make_tensor("value", onnx.TensorProto.FLOAT,
                                               [1], [1])
        node = onnx.helper.make_node(
            'ConstantOfShape',
            inputs=['x'],
            outputs=['y'],
            value=tensor_value,
        )

        y = np.ones(x, dtype=np.float32)
        expect(node, inputs=[x], outputs=[y],
               name='test_constantofshape_float_ones')

    @staticmethod
    def export_int32_zeros() -> None:
        x = np.array([10, 6]).astype(np.int64)
        tensor_value = onnx.helper.make_tensor("value", onnx.TensorProto.INT32,
                                               [1], [0])
        node = onnx.helper.make_node(
            'ConstantOfShape',
            inputs=['x'],
            outputs=['y'],
            value=tensor_value,
        )
        y = np.zeros(x, dtype=np.int32)
        expect(node, inputs=[x], outputs=[y],
               name='test_constantofshape_int_zeros')

    @staticmethod
    def export_int32_shape_zero() -> None:
        x = np.array([0, ]).astype(np.int64)
        tensor_value = onnx.helper.make_tensor("value", onnx.TensorProto.INT32,
                                               [1], [0])
        node = onnx.helper.make_node(
            'ConstantOfShape',
            inputs=['x'],
            outputs=['y'],
            value=tensor_value,
        )
        y = np.zeros(x, dtype=np.int32)
        expect(node, inputs=[x], outputs=[y],
               name='test_constantofshape_int_shape_zero')
