from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect
from typing import List


def unfoldtodepth_2d_reference_implementation(x, shape, padding=[0, 0, 0, 0], stride=[1, 1]):
    # type: (np.ndarray, List[int], List[int], List[int]) -> np.ndarray
    x = np.pad(x, ((0, 0), (0, 0), padding[-4:-2], padding[-2:]), 'constant')
    s0, s1 = x.strides[-2:]
    n_rows = int((x.shape[-2] - shape[0]) / stride[0]) + 1
    n_cols = int((x.shape[-1] - shape[1]) / stride[1]) + 1
    new_shape = n_rows, n_cols, shape[0], shape[1]
    view = np.lib.stride_tricks.as_strided(x, shape=new_shape, strides=(stride[0] * s0, stride[1] * s1, s0, s1))
    return np.transpose(view.reshape(-1, n_rows * n_cols, shape[0] * shape[1]), (0, 2, 1))


class UnfoldToDepth(Base):
    @staticmethod
    def export():  # type: () -> None
        x = np.array([[[[0., 1., 2., 3., 4.],
                        [5., 6., 7., 8., 9.],
                        [10., 11., 12., 13., 14.],
                        [15., 16., 17., 18., 19.],
                        [20., 21., 22., 23., 24.]]]]).astype(np.float32)

        # UnfoldToDepth without padding
        node_without_padding = onnx.helper.make_node(
            'UnfoldToDepth',
            inputs=['x'],
            outputs=['y'],
            block_size=[3, 3],
            # Default values for other attributes: strides=[1, 1], dilations=[1, 1]
            pads=[0, 0, 0, 0],
        )
        y_without_padding = unfoldtodepth_2d_reference_implementation(x, shape=[3, 3])
        # expected [[[0., 1., 2., 5., 6., 7., 10., 11., 12.],  # (1, 9, 9) output tensor
        #            [1., 2., 3., 6., 7., 8., 11., 12., 13.],
        #            [2., 3., 4., 7., 8., 9., 12., 13., 14.],
        #            [5., 6., 7., 10., 11., 12., 15., 16., 17.],
        #            [6., 7., 8., 11., 12., 13., 16., 17., 18.],
        #            [7., 8., 9., 12., 13., 14., 17., 18., 19.],
        #            [10., 11., 12., 15., 16., 17., 20., 21., 22.],
        #            [11., 12., 13., 16., 17., 18., 21., 22., 23.],
        #            [12., 13., 14., 17., 18., 19., 22., 23., 24.]]]

        expect(node_without_padding, inputs=[x], outputs=[y_without_padding],
               name='test_unfoldtodepth_without_padding')

        # UnfoldToDepth with padding
        node_without_padding = onnx.helper.make_node(
            'UnfoldToDepth',
            inputs=['x'],
            outputs=['y'],
            block_size=[3, 3],
            # Default values for other attributes: strides=[1, 1], dilations=[1, 1]
            pads=[1, 1, 1, 1],
        )
        y_with_padding = unfoldtodepth_2d_reference_implementation(x, shape=[3, 3], padding=[1, 1, 1, 1])
        # expected [[[0., 0., 0., 0., 0., 0., 0., 1., 2., 3., 0., 5., 6., 7.,  # (1, 9, 25) output tensor
        #             8., 0., 10., 11., 12., 13., 0., 15., 16., 17., 18.],
        #            [0., 0., 0., 0., 0., 0., 1., 2., 3., 4., 5., 6., 7., 8.,
        #             9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19.],
        #            [0., 0., 0., 0., 0., 1., 2., 3., 4., 0., 6., 7., 8., 9.,
        #             0., 11., 12., 13., 14., 0., 16., 17., 18., 19., 0.],
        #            [0., 0., 1., 2., 3., 0., 5., 6., 7., 8., 0., 10., 11., 12.,
        #             3., 0., 15., 16., 17., 18., 0., 20., 21., 22., 23.],
        #            [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13.,
        #             4., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24.],
        #            [1., 2., 3., 4., 0., 6., 7., 8., 9., 0., 11., 12., 13., 14.,
        #             0., 16., 17., 18., 19., 0., 21., 22., 23., 24., 0.],
        #            [0., 5., 6., 7., 8., 0., 10., 11., 12., 13., 0., 15., 16., 17.,
        #             8., 0., 20., 21., 22., 23., 0., 0., 0., 0., 0.],
        #            [5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18.,
        #             9., 20., 21., 22., 23., 24., 0., 0., 0., 0., 0.],
        #            [6., 7., 8., 9., 0., 11., 12., 13., 14., 0., 16., 17., 18., 19.,
        #             0., 21., 22., 23., 24., 0., 0., 0., 0., 0., 0.]]]

        expect(node_without_padding, inputs=[x], outputs=[y_with_padding],
               name='test_unfoldtodepth_with_padding')

        # UnfoldToDepth with padding and strides
        node_without_padding = onnx.helper.make_node(
            'UnfoldToDepth',
            inputs=['x'],
            outputs=['y'],
            block_size=[3, 3],
            # Default values for other attributes: dilations=[1, 1]
            pads=[2, 2, 2, 2],
            strides=[3, 3]
        )
        y_with_padding = unfoldtodepth_2d_reference_implementation(x, shape=[3, 3], padding=[2, 2, 2, 2], stride=[3, 3])
        # expected [[[ 0.,  0.,  0.,  0.,  6.,  9.,  0., 21., 24.],  # (1, 9, 9) output tensor
        #            [ 0.,  0.,  0.,  0.,  7.,  0.,  0., 22.,  0.],
        #            [ 0.,  0.,  0.,  5.,  8.,  0., 20., 23.,  0.],
        #            [ 0.,  0.,  0.,  0., 11., 14.,  0.,  0.,  0.],
        #            [ 0.,  0.,  0.,  0., 12.,  0.,  0.,  0.,  0.],
        #            [ 0.,  0.,  0., 10., 13.,  0.,  0.,  0.,  0.],
        #            [ 0.,  1.,  4.,  0., 16., 19.,  0.,  0.,  0.],
        #            [ 0.,  2.,  0.,  0., 17.,  0.,  0.,  0.,  0.],
        #            [ 0.,  3.,  0., 15., 18.,  0.,  0.,  0.,  0.]]]

        expect(node_without_padding, inputs=[x], outputs=[y_with_padding],
               name='test_unfoldtodepth_with_padding_stride')
