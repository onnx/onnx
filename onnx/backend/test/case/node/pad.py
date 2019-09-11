from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


def pad_impl(data, pad_width, mode, constant_values=0.0):
    # type: (np.ndarray, tuple, str, float) -> np.ndarray
    if mode == 'constant':
        y = np.pad(
            data,
            pad_width=pad_width,
            mode=mode,
            constant_values=constant_values,
        )
        return y

    y = np.pad(
        data,
        pad_width=pad_width,
        mode=mode,
    )
    return y


class Pad(Base):

    @staticmethod
    def export_constant_pad_with_1D_pads():  # type: () -> None
        node = onnx.helper.make_node(
            'Pad',
            inputs=['x', 'pads', 'value'],
            outputs=['y'],
            mode='constant'
        )
        x = np.random.randn(1, 3, 4, 5).astype(np.float32)
        pads = np.array([0, 0, 1, 3, 0, 0, 2, 4]).astype(np.int64)  # pad order [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
        value = np.array([1.2]).astype(np.float32)
        y = pad_impl(
            x,
            ((0, 0), (0, 0), (1, 2), (3, 4)),  # re-order to np.pad accepted order ((x1_begin, x1_end), (x2_begin, x2_end), ...)
            'constant',
            1.2
        )

        expect(node, inputs=[x, pads, value], outputs=[y],
               name='test_constant_pad_with_1D_pads')

    @staticmethod
    def export_constant_pad_with_2D_pads():  # type: () -> None
        node = onnx.helper.make_node(
            'Pad',
            inputs=['x', 'pads', 'value'],
            outputs=['y'],
            mode='constant'
        )
        x = np.random.randn(1, 3, 4, 5).astype(np.float32)
        pads = np.array([[0, 0, 1, 3, 0, 0, 2, 4]]).astype(np.int64)  # pad order [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
        value = np.array([1.2]).astype(np.float32)
        y = pad_impl(
            x,
            ((0, 0), (0, 0), (1, 2), (3, 4)),  # re-order to np.pad accepted order ((x1_begin, x1_end), (x2_begin, x2_end), ...)
            'constant',
            1.2
        )

        expect(node, inputs=[x, pads, value], outputs=[y],
               name='test_constant_pad_with_2D_pads')

    @staticmethod
    def export_reflection_and_edge_pad_with_1D_pads():  # type: () -> None
        for mode in ['edge', 'reflect']:
            node = onnx.helper.make_node(
                'Pad',
                inputs=['x', 'pads'],
                outputs=['y'],
                mode=mode
            )
            x = np.random.randn(1, 3, 4, 5).astype(np.float32)
            pads = np.array([0, 0, 1, 1, 0, 0, 1, 1]).astype(np.int64)  # pad order [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
            y = pad_impl(
                x,
                ((0, 0), (0, 0), (1, 1), (1, 1)),  # re-order to np.pad accepted order ((x1_begin, x1_end), (x2_begin, x2_end), ...)
                mode
            )

            expect(node, inputs=[x, pads], outputs=[y],
                   name='test_{}_pad_with_1D_pads'.format(mode))
