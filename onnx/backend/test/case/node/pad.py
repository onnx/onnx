# SPDX-License-Identifier: Apache-2.0

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


def pad_impl(data, raw_pads, mode, constant_values=0.0):  # type: ignore

    input_rank = data.ndim
    if input_rank * 2 != raw_pads.size:
        raise Exception('The number of elements in raw_pads should be 2 * data_rank')

    # re-order to np.pad accepted order ((x1_begin, x1_end), (x2_begin, x2_end), ...)
    pad_width = ()
    for i in range(int(raw_pads.size / 2)):
        pad_width += ((raw_pads[i], raw_pads[i + input_rank])),  # type: ignore

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
    def export_constant_pad() -> None:
        node = onnx.helper.make_node(
            'Pad',
            inputs=['x', 'pads', 'value'],
            outputs=['y'],
            mode='constant'
        )
        x = np.random.randn(1, 3, 4, 5).astype(np.float32)
        pads = np.array([0, 0, 1, 3, 0, 0, 2, 4]).astype(np.int64)  # pad order [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
        value = np.float32(1.2)
        y = pad_impl(
            x,
            pads,
            'constant',
            1.2
        )

        expect(node, inputs=[x, pads, value], outputs=[y],
               name='test_constant_pad')

    @staticmethod
    def export_reflection_and_edge_pad() -> None:
        for mode in ['edge', 'reflect']:
            node = onnx.helper.make_node(
                'Pad',
                inputs=['x', 'pads'],
                outputs=['y'],
                mode=mode
            )
            x = np.random.randn(1, 3, 4, 5).astype(np.int32)
            pads = np.array([0, 0, 1, 1, 0, 0, 1, 1]).astype(np.int64)  # pad order [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
            y = pad_impl(
                x,
                pads,
                mode
            )

            expect(node, inputs=[x, pads], outputs=[y],
                   name=f'test_{mode}_pad')
