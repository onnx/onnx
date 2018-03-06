from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import onnx
from ..base import Base
from . import expect


class AveragePool(Base):

    @staticmethod
    def export():
        #2D Test
        #Try different paddings
        for strides in [[1, 1], [2, 2], [3, 1]]:
            #Try different kernels
            for kernel_shape in [[1, 1], [2, 2], [1, 3]]:
                #Try different paddings
                if kernel_shape[1] == 1: paddings = [[0,0,0,0]]
                elif kernel_shape[1] == 2: paddings = [[0,0,0,0], [1,1,1,1], [1,1,0,0]]
                else: paddings = [[0,0,0,0], [0,0,1,1], [0,0,2,2]]
                for pads in paddings:
                    # Define a random input tensor
                    x = np.random.randn(4, 3, 11, 11).astype(np.float32)
                    # Add padding to the input tensor
                    if pads[0] != 0: 
                        x = np.concatenate([np.zeros((x.shape[0], x.shape[1], 
                                pads[0], x.shape[3])), x], 2)
                    if pads[2] != 0:
                        x = np.concatenate([x, np.zeros((x.shape[0], x.shape[1], 
                                pads[2], x.shape[3]))], 2)
                    if pads[1] != 0:
                        x = np.concatenate([np.zeros((x.shape[0], x.shape[1], 
                                x.shape[2], pads[1])), x], 3)
                    if pads[3] != 0:
                        x = np.concatenate([x, np.zeros((x.shape[0], x.shape[1], 
                                x.shape[2], pads[3]))], 3)
                    # Define output tensor of the right size:
                    hSize = int(np.floor((x.shape[2]-(kernel_shape[0]-1)-1)/strides[0]+1))
                    wSize = int(np.floor((x.shape[3]-(kernel_shape[1]-1)-1)/strides[1]+1))
                    y = np.zeros((4, 3, hSize, wSize), dtype=np.float32)
                    # make onnx mean_pool node
                    node = onnx.helper.make_node(
                        'AveragePool',
                        inputs=['x'],
                        outputs=['y'],
                        kernel_shape = kernel_shape,
                        strides=strides,
                        pads=pads
                    )
                    # calculate max_pool using numpy
                    batch = y.shape[0]
                    channel = y.shape[1]
                    x = x.reshape(-1, x.shape[2], x.shape[3])
                    y = y.reshape(-1, y.shape[2], y.shape[3])
                    # TODO get ride of the unefficient loops, maybe 
                    # using the Generalized Universal Function API from numpy
                    for i in range(y.shape[0]):
                        for j in range(y.shape[1]):
                            for k in range(y.shape[2]):
                                startH = j*strides[0]
                                endH = startH+kernel_shape[0]
                                startW = k*strides[1]
                                endW = startW+kernel_shape[1]
                                y[:,j,k] = np.mean(x[:, startH: endH, startW: endW].reshape((batch*channel, -1)), axis=1)
                    x = x.reshape(batch, channel, x.shape[1], x.shape[2])
                    y = y.reshape(batch, channel, y.shape[1], y.shape[2])
                    # Check result:
                    expect(node, inputs=[x], outputs=[y],
                       name='test_average_pooling_2D_with_pad_%d_%d_%d_%d_kernel_%d_%d_stride_%d_%d' % (
                                pads[0],pads[1], pads[2], pads[3], kernel_shape[0], 
                                kernel_shape[1], strides[0], strides[1]))
        #3D Test
        #Try different strides
        for strides in [[1, 1, 1], [2, 2, 2], [3, 1, 3]]:
            #Try different kernels
            for kernel_shape in [[1, 1, 1], [2, 2, 2], [1, 3, 1]]:
                #Try different paddings
                if kernel_shape[1] == 1: paddings = [[0,0,0,0,0,0]]
                elif kernel_shape[1] == 2: paddings = [[0,0,0,0,0,0], [1,1,1,1,1,1], [1,1,1,0,0,0]]
                else: paddings = [[0,0,0,0,0,0], [0,2,0,0,2,0]]
                for pads in paddings:
                    # Define a random input tensor
                    x = np.random.randn(4, 3, 11, 11, 11).astype(np.float32)
                    # Add padding to the input tensor
                    if pads[0] != 0: 
                        x = np.concatenate([np.zeros((x.shape[0], x.shape[1], 
                                pads[0], x.shape[3], x.shape[4])), x], 2)
                    if pads[3] != 0:
                        x = np.concatenate([x, np.zeros((x.shape[0], x.shape[1], 
                                pads[2], x.shape[3], x.shape[4]))], 2)
                    if pads[1] != 0:
                        x = np.concatenate([np.zeros((x.shape[0], x.shape[1], 
                                x.shape[2], pads[1], x.shape[4])), x], 3)
                    if pads[4] != 0:
                        x = np.concatenate([x, np.zeros((x.shape[0], x.shape[1], 
                                x.shape[2], pads[3], x.shape[4]))], 3)
                    if pads[2] != 0:
                        x = np.concatenate([np.zeros((x.shape[0], x.shape[1], 
                                x.shape[2], x.shape[3], pads[2])), x], 4)
                    if pads[5] != 0:
                        x = np.concatenate([x, np.zeros((x.shape[0], x.shape[1], 
                                x.shape[2], x.shape[3], pads[5]))], 4)
                    # Define output tensor of the right size:
                    if strides[0] == 1: y = np.zeros_like(x)
                    # Define output tensor of the right size:
                    hSize = int(np.floor((x.shape[2]-(kernel_shape[0]-1)-1)/strides[0]+1))
                    wSize = int(np.floor((x.shape[3]-(kernel_shape[1]-1)-1)/strides[1]+1))
                    dSize = int(np.floor((x.shape[4]-(kernel_shape[2]-1)-1)/strides[2]+1))
                    y = np.zeros((4, 3, hSize, wSize, dSize), dtype=np.float32)
                    # make onnx max_pool node
                    node = onnx.helper.make_node(
                        'AveragePool',
                        inputs=['x'],
                        outputs=['y'],
                        kernel_shape = kernel_shape,
                        strides=strides,
                        pads=pads
                    )
                    # calculate max_pool using numpy
                    batch = y.shape[0]
                    channel = y.shape[1]
                    x = x.reshape(-1, x.shape[2], x.shape[3], x.shape[4])
                    y = y.reshape(-1, y.shape[2], y.shape[3], y.shape[4])
                    # TODO get ride of the unefficient loops, maybe 
                    # using the Generalized Universal Function API from numpy
                    for j in range(y.shape[1]):
                        for k in range(y.shape[2]):
                            for m in range(y.shape[3]):
                               startH = j*strides[0]
                               endH = startH+kernel_shape[0]
                               startW = k*strides[1]
                               endW = startW+kernel_shape[1]
                               startD = m*strides[2]
                               endD = startD+kernel_shape[2]
                               y[:,j,k,m] = np.mean(x[:, startH: endH, startW: endW, startD: endD].reshape((batch*channel, -1)), axis=1)
                    x = x.reshape(batch, channel, x.shape[1], x.shape[2], x.shape[3])
                    y = y.reshape(batch, channel, y.shape[1], y.shape[2], y.shape[3])
                    # Check result:
                    expect(node, inputs=[x], outputs=[y],
                       name='test_average_pooling_3D_with_pad_%d_%d_%d_%d_%d_%d_kernel_%d_%d_%d_stride_%d_%d_%d' % (
                                pads[0],pads[1], pads[2], pads[3], pads[4], pads[5], kernel_shape[0], 
                                kernel_shape[1], kernel_shape[2], strides[0], strides[1], strides[2]))
