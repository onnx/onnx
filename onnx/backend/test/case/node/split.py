from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import onnx
from ..base import Base
from . import expect


class Split(Base):

    @staticmethod
    def export():
        shape = (6,6)
        test_cases = {
            'num_splits_1': [2],
            'num_splits_2': [3],
            'size_splits_1':[6],
            'size_splits_2':[2,4],
            'size_splits_3':[1,2,3]
            }

        input = np.random.random_sample(shape).astype(np.float32)

        for i in range(len(shape)):
            for test_case, s in test_cases.items():
                output_args = ['output' + str(k) for k in range(len(s) if isinstance(s, list) else s)]
                node = onnx.helper.make_node(
                    'Split',
                    inputs=['input'],
                    outputs=[arg for arg in output_args],
                    axis=i,
                    split=s
                )

                outputs = np.split(input, np.cumsum(s[:-1]) if isinstance(s,list) else s , i)
                expect(node, inputs=[input], outputs=[output for output in outputs],
                name='test_split_' + test_case + '_axis_' + str(i))