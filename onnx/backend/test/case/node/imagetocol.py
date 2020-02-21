from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect
from typing import List, Tuple


def im2col_2d_reference_implementation(x, shape, padding=(0, 0)):  # type: (np.ndarray, List[int], Tuple[int]) -> np.ndarray
    x = np.pad(x, padding)
    s0, s1 = x.strides[-2:]
    nrows = x.shape[-2] - shape[0] + 1
    ncols = x.shape[-1] - shape[1] + 1
    new_shape = shape[0], shape[1], nrows, ncols
    view = np.lib.stride_tricks.as_strided(x, shape=new_shape, strides=(s0, s1, s0, s1))
    return view.reshape(shape[0] * shape[1], -1)[:, :]


class ImageToCol(Base):
    @staticmethod
    def export():  # type: () -> None
        x = np.array([[[0., 1., 2., 3., 4.],
                       [5., 6., 7., 8., 9.],
                       [10., 11., 12., 13., 14.],
                       [15., 16., 17., 18., 19.],
                       [20., 21., 22., 23., 24.]]]).astype(np.float32)

        # ImageToCol without padding
        node_without_padding = onnx.helper.make_node(
            'ImageToCol',
            inputs=['x'],
            outputs=['y'],
            block_shape=[3, 3],
            # Default values for other attributes: strides=[1, 1], dilations=[1, 1]
            pads=[0, 0],
        )
        y_without_padding = im2col_2d_reference_implementation(x[0], [3, 3])
        # expected [[0., 1., 2., 5., 6., 7., 10., 11., 12.],  # (1, 9, 9) output tensor
        #           [1., 2., 3., 6., 7., 8., 11., 12., 13.],
        #           [2., 3., 4., 7., 8., 9., 12., 13., 14.],
        #           [5., 6., 7., 10., 11., 12., 15., 16., 17.],
        #           [6., 7., 8., 11., 12., 13., 16., 17., 18.],
        #           [7., 8., 9., 12., 13., 14., 17., 18., 19.],
        #           [10., 11., 12., 15., 16., 17., 20., 21., 22.],
        #           [11., 12., 13., 16., 17., 18., 21., 22., 23.],
        #           [12., 13., 14., 17., 18., 19., 22., 23., 24.]]

        expect(node_without_padding, inputs=[x], outputs=[y_without_padding],
               name='test_imagetocol_without_padding')

        # ImageToCol with padding
        node_without_padding = onnx.helper.make_node(
            'ImageToCol',
            inputs=['x'],
            outputs=['y'],
            block_shape=[3, 3],
            # Default values for other attributes: strides=[1, 1], dilations=[1, 1], groups=1
            pads=[1, 1, 1, 1],
        )
        y_with_padding = im2col_2d_reference_implementation(x[0], [3, 3], (1, 1,))
        # expected [[0., 0., 0., 0., 0., 0., 0., 1., 2., 3., 0., 5., 6., 7.,
        #            8., 0., 10., 11., 12., 13., 0., 15., 16., 17., 18.],
        #           [0., 0., 0., 0., 0., 0., 1., 2., 3., 4., 5., 6., 7., 8.,
        #            9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19.],
        #           [0., 0., 0., 0., 0., 1., 2., 3., 4., 0., 6., 7., 8., 9.,
        #            0., 11., 12., 13., 14., 0., 16., 17., 18., 19., 0.],
        #           [0., 0., 1., 2., 3., 0., 5., 6., 7., 8., 0., 10., 11., 12.,
        #            3., 0., 15., 16., 17., 18., 0., 20., 21., 22., 23.],
        #           [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13.,
        #            4., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24.],
        #           [1., 2., 3., 4., 0., 6., 7., 8., 9., 0., 11., 12., 13., 14.,
        #            0., 16., 17., 18., 19., 0., 21., 22., 23., 24., 0.],
        #           [0., 5., 6., 7., 8., 0., 10., 11., 12., 13., 0., 15., 16., 17.,
        #            8., 0., 20., 21., 22., 23., 0., 0., 0., 0., 0.],
        #           [5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18.,
        #            9., 20., 21., 22., 23., 24., 0., 0., 0., 0., 0.],
        #           [6., 7., 8., 9., 0., 11., 12., 13., 14., 0., 16., 17., 18., 19.,
        #            0., 21., 22., 23., 24., 0., 0., 0., 0., 0., 0.]]

        expect(node_without_padding, inputs=[x], outputs=[y_with_padding],
               name='test_imagetocol_with_padding')
