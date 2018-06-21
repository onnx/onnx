from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class Hardtanh(Base):

    @staticmethod
    def export_hardtanh_default():  # type: () -> None
        node = onnx.helper.make_node(
            'Hardtanh',
            inputs=['x'],
            outputs=['y'],
        )

        x = np.array([-1, 0, 1]).astype(np.float32)
        y = np.array([-1, 0, 1]).astype(np.float32)
        expect(node, inputs=[x], outputs=[y],
               name='test_hardtanh_default_inbounds')

        x = np.array([-1.1, 0, 1.1]).astype(np.float32)
        y = np.array([-1, 0, 1]).astype(np.float32)
        expect(node, inputs=[x], outputs=[y],
               name='test_hardtanh_default_outbounds')

        x = np.array([-1, 0, 1.1]).astype(np.float32)
        y = np.array([-1, 0, 1]).astype(np.float32)
        expect(node, inputs=[x], outputs=[y],
               name='test_hardtanh_default_splitbounds')

    @staticmethod
    def export_hardtanh_default():  # type: () -> None
        node = onnx.helper.make_node(
            'Hardtanh',
            inputs=['x'],
            outputs=['y'],
            min_val=-5,
            max_val=5,
        )

        x = np.array([-1, 0, 1]).astype(np.float32)
        y = np.array([-1, 0, 1]).astype(np.float32)
        expect(node, inputs=[x], outputs=[y],
               name='test_hardtanh_inbounds')

        x = np.array([-6, 0, 6]).astype(np.float32)
        y = np.array([-5, 0, 5]).astype(np.float32)
        expect(node, inputs=[x], outputs=[y],
               name='test_hardtanh_outbounds')

        x = np.array([-1, 0, 6]).astype(np.float32)
        y = np.array([-1, 0, 5]).astype(np.float32)
        expect(node, inputs=[x], outputs=[y],
               name='test_hardtanh_splitbounds')
