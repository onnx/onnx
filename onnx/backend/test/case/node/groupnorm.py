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
        s = np.array([0, 1, 0, 1]).astype(np.float32)
        
        num_groups = 2
        eps = 1e-05
        
        y = np.array([[[[ 1.0000,  1.0000], [ 1.0000,  1.0000]],
                       [[ 1.8145,  1.8145],[ 3.0724,  1.8145]],
                       [[-1.0000, -1.0000],[-1.0000, -1.0000]],
                       [[ 2.6180,  2.6180],[ 0.8292,  0.3820]]],
                      [[[ 1.0000,  1.0000],[ 1.0000,  1.0000]],
                       [[ 3.7454,  0.4794],[ 0.8876,  0.4794]],
                       [[-1.0000, -1.0000],[-1.0000, -1.0000]],
                       [[ 0.5751,  0.5751],[ 3.0416,  0.5751]]]]).astype(np.float32)

        node = onnx.helper.make_node('GroupNormalization', inputs=['x', 's', 'b'], outputs=['y'], num_groups=num_groups, epsilon=eps)        
        expect(node, inputs=[x, s, b], outputs=[y], name='test_groupnorm')

