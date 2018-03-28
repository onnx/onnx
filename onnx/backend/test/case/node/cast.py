from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import onnx
from onnx import TensorProto
from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE

from ..base import Base
from . import expect


class Cast(Base):

    @staticmethod
    def export():
        shape = (3, 4)
        test_cases = [
            ('FLOAT', 'FLOAT16'),
            ('FLOAT', 'DOUBLE'),
            ('FLOAT16', 'FLOAT'),
            ('FLOAT16', 'DOUBLE'),
            ('DOUBLE', 'FLOAT'),
            ('DOUBLE', 'FLOAT16'),
        ]

        for case in test_cases:
            from_type = case[0]
            to_type = case[1]
            input = np.random.random_sample(shape).astype(
                TENSOR_TYPE_TO_NP_TYPE[getattr(TensorProto, from_type)])
            node = onnx.helper.make_node(
                'Cast',
                inputs=['input'],
                outputs=['output'],
                to=to_type
            )
            output = input.astype(TENSOR_TYPE_TO_NP_TYPE[getattr(TensorProto, to_type)])
            expect(node, inputs=[input], outputs=[output],
                   name='test_cast_' + from_type + '_to_' + to_type)
