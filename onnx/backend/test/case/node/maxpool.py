from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import onnx
from ..base import Base
from . import expect
from .pool_op_common import get_output_shape, get_pad_shape, pool


class MaxPool(Base):

    @staticmethod
    def export_2d_semi_automated_tests():
        #2D Test
        #Try different paddings
        for strides in [[1, 1], [2, 2], [3, 1]]:
            #Try different kernels
            for kernel_shape in [[1, 1], [2, 2], [1, 3]]:
                #Try different paddings
                if kernel_shape[1] == 1:
                    paddings = [[0, 0, 0, 0]]
                elif kernel_shape[1] == 2:
                    paddings = [[1, 1, 1, 1], [1, 0, 1, 0]]
                else:
                    paddings = [[0, 1, 0, 1]]
                for pads in paddings:
                    # Define a random input tensor
                    x = np.random.randn(4, 3, 11, 11).astype(np.float32)
                    # Add padding to the input tensor
                    padded = np.pad(x, ((0, 0), (0, 0), (pads[0], pads[2]), (pads[1], pads[3])), mode='constant',
                                    constant_values=np.nan)
                    # make onnx mean_pool node
                    node = onnx.helper.make_node(
                        'MaxPool',
                        inputs=['x'],
                        outputs=['y'],
                        kernel_shape=kernel_shape,
                        strides=strides,
                        pads=pads
                    )
                    pad_shape = [pads[0] + pads[2], pads[1] + pads[3]]
                    wSize = int(np.floor((padded.shape[2] - (kernel_shape[0] - 1) - 1) / strides[0] + 1))
                    hSize = int(np.floor((padded.shape[3] - (kernel_shape[1] - 1) - 1) / strides[1] + 1))
                    y = pool(padded, np.shape(x), kernel_shape, strides, [wSize, hSize], pad_shape, 'MAX')
                    padStr = 'asym_pad'
                    if pads[0] == pads[2] and pads[1] == pads[3]:
                        padStr = 'sym_pad'
                    elif np.all(pads == 0):
                        padStr = 'no_pad'
                    # Check result:
                    expect(
                        node, inputs=[x], outputs=[y],
                        name='test_maxpool_2D_%s_kernel_%d_%d_stride_%d_%d' % (
                            padStr, kernel_shape[0],
                            kernel_shape[1], strides[0], strides[1]
                        )
                    )

    @staticmethod
    def export_3d_semi_automated_tests():
        #3D Test
        #Try different strides
        for strides in [[1, 1, 1], [2, 2, 2], [3, 1, 3]]:
            #Try different kernels
            for kernel_shape in [[1, 1, 1], [2, 2, 2], [1, 3, 1]]:
                #Try different paddings
                if kernel_shape[1] == 1:
                    paddings = [[0, 0, 0, 0, 0, 0]]
                elif kernel_shape[1] == 2:
                    paddings = [[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0]]
                else:
                    paddings = [[0, 2, 0, 0, 2, 0]]
                for pads in paddings:
                    # Define a random input tensor
                    x = np.random.randn(4, 3, 11, 11, 11).astype(np.float32)
                    # Add padding to the input tensor
                    padded = np.pad(x, ((0, 0), (0, 0), (pads[0], pads[3]), (pads[1], pads[4]), (pads[2], pads[5])), mode='constant',
                                    constant_values=np.nan)
                    # make onnx max_pool node
                    node = onnx.helper.make_node(
                        'MaxPool',
                        inputs=['x'],
                        outputs=['y'],
                        kernel_shape=kernel_shape,
                        strides=strides,
                        pads=pads
                    )
                    pad_shape = [pads[0] + pads[3], pads[1] + pads[4], pads[2] + pads[5]]
                    # Define output tensor of the right size:
                    wSize = int(np.floor((padded.shape[2] - (kernel_shape[0] - 1) - 1) / strides[0] + 1))
                    hSize = int(np.floor((padded.shape[3] - (kernel_shape[1] - 1) - 1) / strides[1] + 1))
                    dSize = int(np.floor((padded.shape[4] - (kernel_shape[2] - 1) - 1) / strides[2] + 1))
                    y = pool(padded, np.shape(x), kernel_shape, strides, [wSize, hSize, dSize], pad_shape, 'MAX')
                    # Check result:
                    padStr = 'asym_pad'
                    if pads[0] == pads[3] and pads[1] == pads[4] and pads[2] == pads[5]:
                        padStr = 'sym_pad'
                    elif np.all(pads == 0):
                        padStr = 'no_pad'
                    expect(
                        node, inputs=[x], outputs=[y],
                        name='test_maxpool_3D_%s_kernel_%d_%d_%d_stride_%d_%d_%d' % (
                            padStr, kernel_shape[0], kernel_shape[1], kernel_shape[2],
                            strides[0], strides[1], strides[2]
                        )
                    )

    @staticmethod
    def export_maxpool_2d_precomputed_pads():
        """
        input_shape: [1, 1, 5, 5]
        output_shape: [1, 1, 5, 5]
        pad_shape: [4, 4] -> [2, 2, 2, 2] by axis
        """
        node = onnx.helper.make_node(
            'MaxPool',
            inputs=['x'],
            outputs=['y'],
            kernel_shape=[5, 5],
            pads=[2, 2, 2, 2]

        )
        x = np.array([[[
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25],
        ]]]).astype(np.float32)
        y = np.array([[[
            [13, 14, 15, 15, 15],
            [18, 19, 20, 20, 20],
            [23, 24, 25, 25, 25],
            [23, 24, 25, 25, 25],
            [23, 24, 25, 25, 25]]]]).astype(np.float32)

        expect(node, inputs=[x], outputs=[y], name='test_maxpool_2d_precomputed_pads')

    @staticmethod
    def export_maxpool_2d_precomputed_strides():
        """
        input_shape: [1, 1, 5, 5]
        output_shape: [1, 1, 2, 2]
        """
        node = onnx.helper.make_node(
            'MaxPool',
            inputs=['x'],
            outputs=['y'],
            kernel_shape=[2, 2],
            strides=[2, 2]
        )
        x = np.array([[[
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25],
        ]]]).astype(np.float32)
        y = np.array([[[[7, 9],
                        [17, 19]]]]).astype(np.float32)

        expect(node, inputs=[x], outputs=[y], name='test_maxpool_2d_precomputed_strides')

    @staticmethod
    def export_maxpool_2d_precomputed_same_upper():
        """
        input_shape: [1, 1, 5, 5]
        output_shape: [1, 1, 3, 3]
        pad_shape: [2, 2] -> [1, 1, 1, 1] by axis
        """
        node = onnx.helper.make_node(
            'MaxPool',
            inputs=['x'],
            outputs=['y'],
            kernel_shape=[3, 3],
            strides=[2, 2],
            auto_pad='SAME_UPPER'
        )
        x = np.array([[[
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25],
        ]]]).astype(np.float32)
        y = np.array([[[[7, 9, 10],
                        [17, 19, 20],
                        [22, 24, 25]]]]).astype(np.float32)

        expect(node, inputs=[x], outputs=[y], name='test_maxpool_2d_precomputed_same_upper')

    @staticmethod
    def export_maxpool_1d_default():
        """
        iutput_shape: [1, 3, 32]
        output_shape: [1, 3, 31]
        """
        node = onnx.helper.make_node(
            'MaxPool',
            inputs=['x'],
            outputs=['y'],
            kernel_shape=[2],
        )
        x = np.random.randn(1, 3, 32).astype(np.float32)
        x_shape = np.shape(x)
        kernel_shape = [2]
        strides = [1]
        out_shape = get_output_shape('VALID', x_shape[2:], kernel_shape, strides)
        padded = x
        y = pool(padded, x_shape, kernel_shape, strides, out_shape, [0], 'MAX')

        expect(node, inputs=[x], outputs=[y], name='test_maxpool_1d_default')

    @staticmethod
    def export_maxpool_2d_default():
        """
        iutput_shape: [1, 3, 32, 32]
        output_shape: [1, 3, 31, 31]
        """
        node = onnx.helper.make_node(
            'MaxPool',
            inputs=['x'],
            outputs=['y'],
            kernel_shape=[2, 2],
        )
        x = np.random.randn(1, 3, 32, 32).astype(np.float32)
        x_shape = np.shape(x)
        kernel_shape = (2, 2)
        strides = (1, 1)
        out_shape = get_output_shape('VALID', x_shape[2:], kernel_shape, strides)
        padded = x
        y = pool(padded, x_shape, kernel_shape, strides, out_shape, (0, 0), 'MAX')

        expect(node, inputs=[x], outputs=[y], name='test_maxpool_2d_default')

    @staticmethod
    def export_maxpool_3d_default():
        """
        iutput_shape: [1, 3, 32, 32, 32]
        output_shape: [1, 3, 31, 31, 31]
        """
        node = onnx.helper.make_node(
            'MaxPool',
            inputs=['x'],
            outputs=['y'],
            kernel_shape=[2, 2, 2],
        )
        x = np.random.randn(1, 3, 32, 32, 32).astype(np.float32)
        x_shape = np.shape(x)
        kernel_shape = [2, 2, 2]
        strides = [1, 1, 1]
        out_shape = get_output_shape('VALID', x_shape[2:], kernel_shape, strides)
        padded = x
        y = pool(padded, x_shape, kernel_shape, strides, out_shape, [0, 0, 0], 'MAX')

        expect(node, inputs=[x], outputs=[y], name='test_maxpool_3d_default')

    @staticmethod
    def export_maxpool_2d_same_upper():
        """
        iutput_shape: [1, 3, 32, 32]
        output_shape: [1, 3, 32, 32]
        pad_shape: [1, 1] -> [0, 1, 0, 1] by axis
        """
        node = onnx.helper.make_node(
            'MaxPool',
            inputs=['x'],
            outputs=['y'],
            kernel_shape=[2, 2],
            auto_pad='SAME_UPPER'
        )
        x = np.random.randn(1, 3, 32, 32).astype(np.float32)
        x_shape = np.shape(x)
        kernel_shape = (2, 2)
        strides = (1, 1)
        out_shape = get_output_shape('SAME_UPPER', x_shape[2:], kernel_shape, strides)
        pad_shape = get_pad_shape('SAME_UPPER', x_shape[2:], kernel_shape, strides, out_shape)
        pad_top = pad_shape[0] // 2
        pad_bottom = pad_shape[0] - pad_top
        pad_left = pad_shape[1] // 2
        pad_right = pad_shape[1] - pad_left
        padded = np.pad(x, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant',
                        constant_values=np.nan)
        y = pool(padded, x_shape, kernel_shape, strides, out_shape, pad_shape, 'MAX')

        expect(node, inputs=[x], outputs=[y], name='test_maxpool_2d_same_upper')

    @staticmethod
    def export_maxpool_2d_same_lower():
        """
        iutput_shape: [1, 3, 32, 32]
        output_shape: [1, 3, 32, 32]
        pad_shape: [1, 1] -> [1, 0, 1, 0] by axis
        """
        node = onnx.helper.make_node(
            'MaxPool',
            inputs=['x'],
            outputs=['y'],
            kernel_shape=[2, 2],
            auto_pad='SAME_LOWER'
        )
        x = np.random.randn(1, 3, 32, 32).astype(np.float32)
        x_shape = np.shape(x)
        kernel_shape = (2, 2)
        strides = (1, 1)
        out_shape = get_output_shape('SAME_LOWER', x_shape[2:], kernel_shape, strides)
        pad_shape = get_pad_shape('SAME_LOWER', x_shape[2:], kernel_shape, strides, out_shape)
        pad_bottom = pad_shape[0] // 2
        pad_top = pad_shape[0] - pad_bottom
        pad_right = pad_shape[1] // 2
        pad_left = pad_shape[1] - pad_right
        padded = np.pad(x, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant',
                        constant_values=np.nan)
        y = pool(padded, x_shape, kernel_shape, strides, out_shape, pad_shape, 'MAX')

        expect(node, inputs=[x], outputs=[y], name='test_maxpool_2d_same_lower')

    @staticmethod
    def export_maxpool_2d_pads():
        """
        iutput_shape: [1, 3, 28, 28]
        output_shape: [1, 3, 30, 30]
        pad_shape: [4, 4] -> [2, 2, 2, 2] by axis
        """
        node = onnx.helper.make_node(
            'MaxPool',
            inputs=['x'],
            outputs=['y'],
            kernel_shape=[3, 3],
            pads=[2, 2, 2, 2]
        )
        x = np.random.randn(1, 3, 28, 28).astype(np.float32)
        x_shape = np.shape(x)
        kernel_shape = (3, 3)
        strides = (1, 1)
        pad_bottom = 2
        pad_top = 2
        pad_right = 2
        pad_left = 2
        pad_shape = [pad_top + pad_bottom, pad_left + pad_right]
        out_shape = get_output_shape('VALID', np.add(x_shape[2:], pad_shape), kernel_shape, strides)
        padded = np.pad(x, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant',
                        constant_values=np.nan)
        y = pool(padded, x_shape, kernel_shape, strides, out_shape, pad_shape, 'MAX')

        expect(node, inputs=[x], outputs=[y], name='test_maxpool_2d_pads')

    @staticmethod
    def export_maxpool_2d_strides():
        """
        iutput_shape: [1, 3, 32, 32]
        output_shape: [1, 3, 10, 10]
        """
        node = onnx.helper.make_node(
            'MaxPool',
            inputs=['x'],
            outputs=['y'],
            kernel_shape=[5, 5],
            strides=[3, 3]
        )
        x = np.random.randn(1, 3, 32, 32).astype(np.float32)
        x_shape = np.shape(x)
        kernel_shape = (5, 5)
        strides = (3, 3)
        out_shape = get_output_shape('VALID', x_shape[2:], kernel_shape, strides)
        padded = x
        y = pool(padded, x_shape, kernel_shape, strides, out_shape, (0, 0), 'MAX')

        expect(node, inputs=[x], outputs=[y], name='test_maxpool_2d_strides')
