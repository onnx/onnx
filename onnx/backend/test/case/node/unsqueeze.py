from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class Unsqueeze(Base):

    @staticmethod
    def export_unsqueeze_one_axis():  # type: () -> None
        x = np.random.randn(3, 4, 5).astype(np.float32)

        for i in range(x.ndim):
            node = onnx.helper.make_node(
                'Unsqueeze',
                inputs=['x'],
                outputs=['y'],
                axes=[i],
            )
            y = np.expand_dims(x, axis=i)

            expect(node, inputs=[x], outputs=[y],
                   name='test_unsqueeze_axis_' + str(i))

    @staticmethod
    def export_unsqueeze_two_axes():  # type: () -> None
        x = np.random.randn(3, 4, 5).astype(np.float32)

        node = onnx.helper.make_node(
            'Unsqueeze',
            inputs=['x'],
            outputs=['y'],
            axes=[1, 4],
        )
        y = np.expand_dims(x, axis=1)
        y = np.expand_dims(y, axis=4)

        expect(node, inputs=[x], outputs=[y],
                name='test_unsqueeze_two_axes')

    @staticmethod
    def export_unsqueeze_three_axes():  # type: () -> None
        x = np.random.randn(3, 4, 5).astype(np.float32)

        node = onnx.helper.make_node(
            'Unsqueeze',
            inputs=['x'],
            outputs=['y'],
            axes=[2, 4, 5],
        )
        y = np.expand_dims(x, axis=2)
        y = np.expand_dims(y, axis=4)
        y = np.expand_dims(y, axis=5)

        expect(node, inputs=[x], outputs=[y],
                name='test_unsqueeze_three_axes')

    @staticmethod
    def export_unsqueeze_unsorted_axes():  # type: () -> None
        x = np.random.randn(3, 4, 5).astype(np.float32)

        node = onnx.helper.make_node(
            'Unsqueeze',
            inputs=['x'],
            outputs=['y'],
            axes=[5, 4, 2],
        )
        y = np.expand_dims(x, axis=2)
        y = np.expand_dims(y, axis=4)
        y = np.expand_dims(y, axis=5)

        expect(node, inputs=[x], outputs=[y],
                name='test_unsqueeze_unsorted_axes')

    @staticmethod
    def export_unsqueeze_negative_axes():  # type: () -> None
        node = onnx.helper.make_node(
            'Unsqueeze',
            inputs=['x'],
            outputs=['y'],
            axes=[-2],
        )
        x = np.random.randn(1, 3, 1, 5).astype(np.float32)
        y = np.expand_dims(x, axis=-2)
        expect(node, inputs=[x], outputs=[y],
               name='test_unsqueeze_negative_axes')
