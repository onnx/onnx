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

        x = np.array([1, 2, 3, 4]).astype(np.float32)
        s = np.array([2]).astype(np.float32)
        b = np.array([-1]).astype(np.float32)
        
        
        #y = x * s + b
        y = np.array([1, 2, 3, 4]).astype(np.float32)

        node = onnx.helper.make_node('GroupNormalization', inputs=['x', 's', 'b'], outputs=['y'])        
        expect(node, inputs=[x, s, b], outputs=[y], name='test_groupnorm_example')

