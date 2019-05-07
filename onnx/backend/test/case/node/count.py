from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class Count(Base):

    @staticmethod
    def export_count_single_input():  # type: () -> None
        node = onnx.helper.make_node(
            'Count',
            inputs=['x'],
            outputs=['uniques', 'counts'],
        )

        x = np.array([1, 2, 3, 3, 2, 1, 1, 1, 2])
        uniques = np.array([1, 2, 3])
        counts = np.array([4, 3, 2])
        expect(node, inputs=[x], outputs=[uniques, counts],
               name='test_count_single_input')

    @staticmethod
    def export_count():  # type: () -> None
        node = onnx.helper.make_node(
            'Count',
            inputs=['x', 'uniques'],
            outputs=['uniques', 'counts'],
        )

        x = np.array([1, 2, 3, 3, 2, 1, 1, 1, 2])
        uniques = np.array([3, 2, 1])
        counts = np.array([2, 3, 4])
        expect(node, inputs=[x, uniques], outputs=[uniques, counts],
               name='test_count')
