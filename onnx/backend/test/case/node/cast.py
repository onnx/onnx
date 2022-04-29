# SPDX-License-Identifier: Apache-2.0
# coding: utf-8


import numpy as np  # type: ignore

import onnx
from onnx import TensorProto
from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE

from ..base import Base
from . import expect
import sys


class Cast(Base):

    @staticmethod
    def export() -> None:
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
            ('FLOAT', 'BFLOAT16'),
            ('BFLOAT16', 'FLOAT'),
        ]

        for from_type, to_type in test_cases:
            input_type_proto = None
            output_type_proto = None
            if 'BFLOAT16' == from_type or 'BFLOAT16' == to_type:
                np_fp32 = np.array([u'0.47892547', u'0.48033667', u'0.49968487', u'0.81910545',
                    u'0.47031248', u'0.816468', u'0.21087195', u'0.7229038',
                    u'NaN', u'INF', u'+INF', u'-INF'], dtype=np.float32)
                little_endisan = sys.byteorder == 'little'
                np_uint16_view = np_fp32.view(dtype=np.uint16)
                np_bfp16 = np_uint16_view[1::2] if little_endisan else np_uint16_view[0::2]
                if 'BFLOAT16' == to_type:
                    assert from_type == 'FLOAT'
                    input = np_fp32.reshape([3, 4])
                    output = np_bfp16.reshape([3, 4])
                    input_type_proto = onnx.helper.make_tensor_type_proto(int(TensorProto.FLOAT), input.shape)
                    output_type_proto = onnx.helper.make_tensor_type_proto(int(TensorProto.BFLOAT16), output.shape)
                else:
                    assert to_type == 'FLOAT'
                    input = np_bfp16.reshape([3, 4])
                    #convert bfloat to FLOAT
                    np_fp32_zeros = np.zeros((len(np_bfp16) * 2,), dtype=np.uint16)
                    if little_endisan:
                        np_fp32_zeros[1::2] = np_bfp16
                    else:
                        np_fp32_zeros[0::2] = np_bfp16
                    np_fp32_from_bfloat = np_fp32_zeros.view(dtype=np.float32)
                    output = np_fp32_from_bfloat.reshape([3, 4])
                    input_type_proto = onnx.helper.make_tensor_type_proto(int(TensorProto.BFLOAT16), input.shape)
                    output_type_proto = onnx.helper.make_tensor_type_proto(int(TensorProto.FLOAT), output.shape)
            elif 'STRING' != from_type:
                input = np.random.random_sample(shape).astype(
                    TENSOR_TYPE_TO_NP_TYPE[getattr(TensorProto, from_type)])
                if ('STRING' == to_type):
                    # Converting input to str, then give it object dtype for generating script
                    ss = []
                    for i in input.flatten():
                        s = str(i).encode('utf-8')
                        su = s.decode('utf-8')
                        ss.append(su)

                    output = np.array(ss).astype(object).reshape([3, 4])
                else:
                    output = input.astype(TENSOR_TYPE_TO_NP_TYPE[getattr(TensorProto, to_type)])
            else:
                input = np.array([u'0.47892547', u'0.48033667', u'0.49968487', u'0.81910545',
                    u'0.47031248', u'0.816468', u'0.21087195', u'0.7229038',
                    u'NaN', u'INF', u'+INF', u'-INF'], dtype=np.dtype(object)).reshape([3, 4])
                output = input.astype(TENSOR_TYPE_TO_NP_TYPE[getattr(TensorProto, to_type)])
            node = onnx.helper.make_node(
                'Cast',
                inputs=['input'],
                outputs=['output'],
                to=getattr(TensorProto, to_type),
            )
            if input_type_proto and output_type_proto:
                expect(node, inputs=[input], outputs=[output],
                           name='test_cast_' + from_type + '_to_' + to_type,
                           input_type_protos=[input_type_proto],
                           output_type_protos=[output_type_proto])
            else:
                expect(node, inputs=[input], outputs=[output],
                           name='test_cast_' + from_type + '_to_' + to_type)
