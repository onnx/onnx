from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class IsInf(Base):

    @staticmethod
    def export_infinity():  # type: () -> None
        node = onnx.helper.make_node('IsInf',
                                     inputs=['x'],
                                     outputs=['y'],
                                     )

        x = np.array([-1.2, np.nan, np.inf, 2.8, np.NINF, np.inf],
                     dtype=np.float32)
        y = np.isinf(x)
        expect(node, inputs=[x], outputs=[y], name='test_isinf')

    @staticmethod
    def export_positive_infinity_only():  # type: () -> None
        node = onnx.helper.make_node('IsInf',
                                     inputs=['x'],
                                     outputs=['y'],
                                     detect_negative=0
                                     )

        x = np.array([-1.7, np.nan, np.inf, 3.6, np.NINF, np.inf],
                     dtype=np.float32)
        y = np.isposinf(x)
        expect(node, inputs=[x], outputs=[y], name='test_isinf_positive')

    @staticmethod
    def export_negative_infinity_only():  # type: () -> None
        node = onnx.helper.make_node('IsInf',
                                     inputs=['x'],
                                     outputs=['y'],
                                     detect_positive=0
                                     )

        x = np.array([-1.7, np.nan, np.inf, -3.6, np.NINF, np.inf],
                     dtype=np.float32)
        y = np.isneginf(x)
        expect(node, inputs=[x], outputs=[y], name='test_isinf_negative')
