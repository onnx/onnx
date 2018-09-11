from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect
from typing import Dict, Sequence, Text, Any


class Concat(Base):

    @staticmethod
    def export():  # type: () -> None
        test_cases = {
            '1d': ([1, 2],
                   [3, 4]),
            '2d': ([[1, 2], [3, 4]],
                   [[5, 6], [7, 8]]),
            '3d': ([[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                   [[[9, 10], [11, 12]], [[13, 14], [15, 16]]])
        }  # type: Dict[Text, Sequence[Any]]

        for test_case, values_ in test_cases.items():
            values = [np.asarray(v, dtype=np.float32) for v in values_]
            for i in range(len(values[0].shape)):
                in_args = ['value' + str(k) for k in range(len(values))]
                node = onnx.helper.make_node(
                    'Concat',
                    inputs=[s for s in in_args],
                    outputs=['output'],
                    axis=i
                )
                output = np.concatenate(values, i)
                expect(node, inputs=[v for v in values], outputs=[output],
                       name='test_concat_' + test_case + '_axis_' + str(i))
