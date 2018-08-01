from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class ConvInteger(Base):

    @staticmethod
    def export():  # type: () -> None

        x = np.array([[[[0, 1, 2, 3, 4],  # (1, 1, 5, 5) input tensor
                        [5, 6, 7, 8, 9],
                        [10, 11, 12, 13, 14],
                        [15, 16, 17, 18, 19],
                        [20, 21, 22, 23, 24]]]]).astype(np.uint8)
        W = np.array([[[[1, 1, 1],  # (1, 1, 3, 3) tensor for convolution weights
                        [1, 1, 1],
                        [1, 1, 1]]]]).astype(np.uint8)
        Z = np.array([1]).astype(np.uint8)  # padding value (zero point).
        # Convolution with padding
        node_with_padding = onnx.helper.make_node(
            'ConvInteger',
            inputs=['x', 'W', 'Z'],
            outputs=['y'],
            # Default values for other attributes: strides=[1, 1], dilations=[1, 1], groups=1
            pads=[1, 1, 1, 1],
        )
        y_with_padding = np.array([[[[17, 24, 30, 36, 29],  # (1, 1, 5, 5) output tensor
                                     [36, 54, 63, 72, 54],
                                     [66, 99, 108, 117, 84],
                                     [96, 144, 153, 162, 114],
                                     [77, 114, 120, 126, 89]]]]).astype(np.uint32)
        expect(node_with_padding, inputs=[x, W, Z], outputs=[y_with_padding],
               name='test_basic_convinteger_with_padding')

        # Convolution without padding
        node_without_padding = onnx.helper.make_node(
            'ConvInteger',
            inputs=['x', 'W', 'Z'],
            outputs=['y'],
            # Default values for other attributes: strides=[1, 1], dilations=[1, 1], groups=1
            pads=[0, 0, 0, 0],
        )
        y_without_padding = np.array([[[[54, 63, 72],  # (1, 1, 3, 3) output tensor
                                        [99, 108, 117],
                                        [144, 153, 162]]]]).astype(np.uint32)
        expect(node_without_padding, inputs=[x, W, Z], outputs=[y_without_padding],
               name='test_basic_convinteger_without_padding')

    @staticmethod
    def export_convinteger_with_strides():  # type: () -> None

        x = np.array([[[[0, 1, 2, 3, 4],  # (1, 1, 7, 5) input tensor
                        [5, 6, 7, 8, 9],
                        [10, 11, 12, 13, 14],
                        [15, 16, 17, 18, 19],
                        [20, 21, 22, 23, 24],
                        [25, 26, 27, 28, 29],
                        [30, 31, 32, 33, 34]]]]).astype(np.uint8)
        W = np.array([[[[1, 1, 1],  # (1, 1, 3, 3) tensor for convolution weights
                        [1, 1, 1],
                        [1, 1, 1]]]]).astype(np.uint8)
        Z = np.array([1]).astype(np.uint8)  # padding value (zero point).
        # Convolution with strides=2 and padding
        node_with_padding = onnx.helper.make_node(
            'ConvInteger',
            inputs=['x', 'W', 'Z'],
            outputs=['y'],
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
            strides=[2, 2],  # Default values for other attributes: dilations=[1, 1], groups=1
        )
        y_with_padding = np.array([[[[17, 30, 29],  # (1, 1, 4, 3) output tensor
                                     [66., 108, 84],
                                     [126, 198, 144],
                                     [117, 180, 129]]]]).astype(np.uint32)
        expect(node_with_padding, inputs=[x, W, Z], outputs=[y_with_padding],
               name='test_convinteger_with_strides_padding')

        # Convolution with strides=2 and no padding
        node_without_padding = onnx.helper.make_node(
            'ConvInteger',
            inputs=['x', 'W', 'Z'],
            outputs=['y'],
            kernel_shape=[3, 3],
            pads=[0, 0, 0, 0],
            strides=[2, 2],  # Default values for other attributes: dilations=[1, 1], groups=1
        )
        y_without_padding = np.array([[[[54, 72],  # (1, 1, 3, 2) output tensor
                                        [144, 162],
                                        [234, 252]]]]).astype(np.uint32)
        expect(node_without_padding, inputs=[x, W, Z], outputs=[y_without_padding],
               name='test_convinteger_with_strides_no_padding')

        # Convolution with strides=2 and padding only along one dimension (the H dimension in NxCxHxW tensor)
        node_with_asymmetric_padding = onnx.helper.make_node(
            'ConvInteger',
            inputs=['x', 'W', 'Z'],
            outputs=['y'],
            kernel_shape=[3, 3],
            pads=[1, 0, 1, 0],
            strides=[2, 2],  # Default values for other attributes: dilations=[1, 1], groups=1
        )
        y_with_asymmetric_padding = np.array([[[[24, 36],  # (1, 1, 4, 2) output tensor
                                                [99, 117],
                                                [189, 207],
                                                [174, 186]]]]).astype(np.uint32)
        expect(node_with_asymmetric_padding, inputs=[x, W, Z], outputs=[y_with_asymmetric_padding],
               name='test_convinteger_with_strides_and_asymmetric_padding')
