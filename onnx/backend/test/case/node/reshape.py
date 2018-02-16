from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import onnx
from ..base import Base
from . import expect


class Reshape(Base):

    @staticmethod
    def export():
        original_shape = [2, 3, 4]
        test_cases = {
            'reordered_dims':[4, 2, 3],
            'reduced_dims':[3, 8],
            'extended_dims':[3, 2, 2, 2],
            'one_dim':[24],
            'negative_dim':[6, -1, 2]
        }
        data = np.random.random_sample(original_shape).astype(np.float32)

        for test_name,test_shape in test_cases.items():
            node = onnx.helper.make_node(
                'Reshape',
                inputs=['data'],
                outputs=['reshaped'],
                shape=test_shape,
            )

            reshaped = np.reshape(data, test_shape)
            expect(node, inputs=[data], outputs=[reshaped],
               name='test_reshape_' + test_name)