from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class Resize(Base):

    @staticmethod
    def export_upsample_nearest():  # type: () -> None
        node = onnx.helper.make_node(
            'Resize',
            inputs=['X', 'scales'],
            outputs=['Y'],
            mode='nearest',
        )

        data = np.array([[[
            [1, 2],
            [3, 4],
        ]]], dtype=np.float32)

        scales = np.array([1.0, 1.0, 2.0, 3.0], dtype=np.float32)

        output = np.array([[[
            [1, 1, 1, 2, 2, 2],
            [1, 1, 1, 2, 2, 2],
            [3, 3, 3, 4, 4, 4],
            [3, 3, 3, 4, 4, 4],
        ]]], dtype=np.float32)

        expect(node, inputs=[data, scales], outputs=[output],
               name='test_resize_upsample_nearest')

    @staticmethod
    def export_downsample_nearest():  # type: () -> None
        node = onnx.helper.make_node(
            'Resize',
            inputs=['X', 'scales'],
            outputs=['Y'],
            mode='nearest',
        )

        data = np.array([[[
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ]]], dtype=np.float32)

        scales = np.array([1.0, 1.0, 0.6, 0.6], dtype=np.float32)

        output = np.array([[[
            [1, 3]
        ]]], dtype=np.float32)

        expect(node, inputs=[data, scales], outputs=[output],
               name='test_resize_downsample_nearest')

    @staticmethod
    def export_upsample_linear():  # type: () -> None
        node = onnx.helper.make_node(
            'Resize',
            inputs=['X', 'scales'],
            outputs=['Y'],
            mode='linear',
        )

        data = np.array([[[
            [1, 2],
            [3, 4],
        ]]], dtype=np.float32)

        scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)

        output = np.array([[[
            [1, 1.5, 2, 2],
            [2, 2.5, 3, 3],
            [3, 3.5, 4, 4],
            [3, 3.5, 4, 4],
        ]]], dtype=np.float32)

        expect(node, inputs=[data, scales], outputs=[output],
               name='test_resize_upsample_linear')

    @staticmethod
    def export_downsample_linear():  # type: () -> None
        node = onnx.helper.make_node(
            'Resize',
            inputs=['X', 'scales'],
            outputs=['Y'],
            mode='linear',
        )

        data = np.array([[[
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ]]], dtype=np.float32)

        scales = np.array([1.0, 1.0, 0.6, 0.6], dtype=np.float32)

        output = np.array([[[
            [1, 2.66666651]
        ]]], dtype=np.float32)

        expect(node, inputs=[data, scales], outputs=[output],
               name='test_resize_downsample_linear')
