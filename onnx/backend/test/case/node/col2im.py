# SPDX-License-Identifier: Apache-2.0

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class Col2Im(Base):

    @staticmethod
    def export() -> None:
        input = np.array([[[1., 6., 11., 16., 21.],  # (1, 5, 5)
                           [2., 7., 12., 17., 22.],
                           [3., 8., 13., 18., 23.],
                           [4., 9., 14., 19., 24.],
                           [5., 0., 15., 20., 25.]]]).astype(np.float32)
        image_shape = np.array([5, 5]).astype(np.int64)
        block_shape = np.array([1, 5]).astype(np.int64)
        node = onnx.helper.make_node(
            "Col2Im",
            ["input", "image_shape", "block_shape"],
            ["output"]
        )

        output = np.array([[[[1., 2., 3., 4., 5.],  # (1, 1, 5, 5)
                             [6., 7., 8., 9., 0.],
                             [11., 12., 13., 14., 15.],
                             [16., 17., 18., 19., 20.],
                             [21., 22., 23., 24., 25.]]]]).astype(np.float32)

        expect(
            node,
            inputs=[input, image_shape, block_shape],
            outputs=[output],
            name='test_col2im'
        )

    @staticmethod
    def export_col2im_strides() -> None:
        input = np.array([[[0., 0., 0., 0.],  # (1, 9, 4)
                           [1., 1., 1., 1.],
                           [1., 1., 1., 1.],
                           [1., 1., 1., 1.],
                           [0., 0., 0., 0.],
                           [0., 0., 0., 0.],
                           [0., 0., 0., 0.],
                           [1., 1., 1., 1.],
                           [0., 0., 0., 0.]]]).astype(np.float32)
        image_shape = np.array([5, 5]).astype(np.int64)
        block_shape = np.array([3, 3]).astype(np.int64)

        output = np.array([[[[0., 1., 1., 1., 1.],  # (1, 1, 5, 5)
                             [1., 0., 1., 0., 0.],
                             [0., 2., 1., 2., 1.],
                             [1., 0., 1., 0., 0.],
                             [0., 1., 0., 1., 0.]]]]).astype(np.float32)

        node = onnx.helper.make_node(
            "Col2Im",
            ["input", "image_shape", "block_shape"],
            ["output"],
            strides=[2, 2]
        )
        expect(
            node,
            inputs=[input, image_shape, block_shape],
            outputs=[output],
            name='test_col2im_strides'
        )

    @staticmethod
    def export_col2im_pads() -> None:
        input = np.array([[[1., 6., 11., 16., 21., 26, 31, 36, 41, 46, 51, 56, 61, 66, 71],  # (1, 5, 15)
                           [2., 7., 12., 17., 22., 27, 32, 37, 42, 47, 52, 57, 62, 67, 72],
                           [3., 8., 13., 18., 23., 28, 33, 38, 43, 48, 53, 58, 63, 68, 73],
                           [4., 9., 14., 19., 24., 29, 34, 39, 44, 49, 54, 59, 64, 69, 74],
                           [5., 10., 15., 20., 25., 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]]]).astype(np.float32)
        image_shape = np.array([5, 5]).astype(np.int64)
        block_shape = np.array([1, 5]).astype(np.int64)

        output = np.array([[[[8., 21., 24., 27., 14.],  # (1, 1, 5, 5)
                             [38., 66., 69., 72., 54.],
                             [68., 111., 114., 117., 84.],
                             [98., 156., 159., 162., 114.],
                             [128., 201., 204., 207., 144.]]]]).astype(np.float32)

        node = onnx.helper.make_node(
            "Col2Im",
            ["input", "image_shape", "block_shape"],
            ["output"],
            pads=[0, 1, 0, 1]
        )
        expect(
            node,
            inputs=[input, image_shape, block_shape],
            outputs=[output],
            name='test_col2im_pads'
        )

    @staticmethod
    def export_col2im_dilations() -> None:
        input = np.array([[[1., 5., 9., 13., 17],  # (1, 4, 5)
                           [2., 6., 10., 14., 18],
                           [3., 7., 11., 15., 19],
                           [4., 8., 12., 16., 20]]]).astype(np.float32)
        image_shape = np.array([6, 6]).astype(np.int64)
        block_shape = np.array([2, 2]).astype(np.int64)

        output = np.array([[[[1., 0., 0., 0., 0., 2.],  # (1, 1, 6, 6)
                             [8., 0., 0., 0., 0., 10.],
                             [16., 0., 0., 0., 0., 18.],
                             [24., 0., 0., 0., 0., 26.],
                             [32., 0., 0., 0., 0., 34.],
                             [19., 0., 0., 0., 0., 20.]]]]).astype(np.float32)

        node = onnx.helper.make_node(
            "Col2Im",
            ["input", "image_shape", "block_shape"],
            ["output"],
            dilations=[1, 5]
        )
        expect(
            node,
            inputs=[input, image_shape, block_shape],
            outputs=[output],
            name='test_col2im_dilations'
        )
