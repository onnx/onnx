# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from onnx import TensorProto
from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE

from ..base import Base
from . import expect


class Cast(Base):

    @staticmethod
    def export():  # type: () -> None
        shape = (3, 4)
        test_cases = [
            ('FLOAT', 'FLOAT16'),
            ('FLOAT', 'DOUBLE'),
            ('FLOAT16', 'FLOAT'),
            ('FLOAT16', 'DOUBLE'),
            ('DOUBLE', 'FLOAT'),
            ('DOUBLE', 'FLOAT16'),
            ('FLOAT', 'STRING'),
            ('STRING', 'FLOAT'),
        ]

        for from_type, to_type in test_cases:
            if 'STRING' != from_type:
                input = np.random.random_sample(shape).astype(
                    TENSOR_TYPE_TO_NP_TYPE[getattr(TensorProto, from_type)])
                if ('STRING' == to_type):
                    # Converting input to str, then give it np.object dtype for generating script
                    ss = []
                    for i in input.flatten():
                        s = str(i).encode('utf-8')
                        su = s.decode('utf-8')
                        ss.append(su)

                    output = np.array(ss).astype(np.object).reshape([3, 4])
                else:
                    output = input.astype(TENSOR_TYPE_TO_NP_TYPE[getattr(TensorProto, to_type)])
            else:
                input = np.array([u'0.47892547', u'0.48033667', u'0.49968487', u'0.81910545',
                    u'0.47031248', u'0.816468', u'0.21087195', u'0.7229038',
                    u'NaN', u'INF', u'+INF', u'-INF'], dtype=np.dtype(np.object)).reshape([3, 4])
                output = input.astype(TENSOR_TYPE_TO_NP_TYPE[getattr(TensorProto, to_type)])
            node = onnx.helper.make_node(
                'Cast',
                inputs=['input'],
                outputs=['output'],
                to=getattr(TensorProto, to_type),
            )
            expect(node, inputs=[input], outputs=[output],
                       name='test_cast_' + from_type + '_to_' + to_type)
