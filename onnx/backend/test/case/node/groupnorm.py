from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class GroupNormalization(Base):

    @staticmethod
    def export():  # type: () -> None

        x = np.array([[[[4., 5.], [0., 6.]],
                       [[6., 6.], [9., 6.]],
                       [[3., 4.], [6., 0.]],
                       [[6., 6.], [2., 1.]]],
                      [[[3., 3.], [5., 4.]],
                       [[9., 1.], [2., 1.]],
                       [[6., 2.], [4., 2.]],
                       [[1., 1.], [6., 1.]]]]).astype(np.float32)

        b = np.array([1.0, 1.5, -1.0, 1.5]).astype(np.float32)
        s = np.array([[0, 1, 0, 1]]).astype(np.float32)
        
        num_groups = 2
        eps = 1e-05

        #y = x * s + b
        y = np.array([[[[-0.5241, -0.1048], [-2.2014,  0.3145]],
                       [[ 1.4717,  1.4717], [ 3.3586,  1.4717]],
                       [[ 0.2236, -0.2236], [-1.1180,  1.5652]],
                       [[ 2.6770,  2.6770], [-0.0062, -0.6770]]],
                      [[[-0.2041, -0.2041], [ 0.6124,  0.2041]],
                       [[ 4.3680, -0.5309], [ 0.0814, -0.5309]],
                       [[-1.5416,  0.4316], [-0.5550,  0.4316]],
                       [[-0.3874, -0.3874], [ 3.3123, -0.3874]]]]).astype(np.float32)

        node = onnx.helper.make_node('GroupNormalization', inputs=['x', 's', 'b'], outputs=['y'])        
        expect(node, inputs=[x, s, b], outputs=[y], name='test_groupnorm_example')

        '''
        def _groupnorm_test_mode(x, s, bias, epsilon=1e-5):  # type: ignore

            #dims_x = len(x.shape)
            #dim_ones = (1,) * (dims_x - 2)
            #s = s.reshape(-1, *dim_ones)
            #var = var.reshape(-1, *dim_ones)
            #return s * (x - mean) / np.sqrt(var + epsilon) + bias
            return x

        # input size: (1, 2, 1, 3)
        x = np.array([[[[-1, 0, 1]], [[2, 3, 4]]]]).astype(np.float32)
        s = np.array([1.0, 1.5]).astype(np.float32)
        bias = np.array([0, 1]).astype(np.float32)
        y = _groupnorm_test_mode(x, s, bias).astype(np.float32)

        node = onnx.helper.make_node(
            'GroupNormalization',
            inputs=['x', 's', 'bias'],
            outputs=['y'],
        )

        # output size: (1, 2, 1, 3)
        expect(node, inputs=[x, s, bias], outputs=[y],
               name='test_groupnorm_example')

        # input size: (2, 3, 4, 5)
        x = np.random.randn(2, 3, 4, 5).astype(np.float32)
        s = np.random.randn(3).astype(np.float32)
        bias = np.random.randn(3).astype(np.float32)
        epsilon = 1e-2
        y = _groupnorm_test_mode(x, s, bias, epsilon).astype(np.float32)

        node = onnx.helper.make_node(
            'GroupNormalization',
            inputs=['x', 's', 'bias'],
            outputs=['y'],
            epsilon=epsilon,
        )

        # output size: (2, 3, 4, 5)
        expect(node, inputs=[x, s, bias], outputs=[y],
               name='test_groupnorm_epsilon')
        '''