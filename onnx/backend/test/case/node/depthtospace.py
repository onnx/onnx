from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class DepthToSpace(Base):

    @staticmethod
    def export():  # type: () -> None
        b, c, h, w = shape = (2, 8, 3, 3)
        blocksize = 2
        node = onnx.helper.make_node(
            'DepthToSpace',
            inputs=['x'],
            outputs=['y'],
            blocksize=blocksize,
        )
        x = np.random.random_sample(shape).astype(np.float32)
        tmp = np.reshape(x, [b, blocksize, blocksize, c // (blocksize**2), h, w])
        tmp = np.transpose(tmp, [0, 3, 4, 1, 5, 2])
        y = np.reshape(tmp, [b, c // (blocksize**2), h * blocksize, w * blocksize])
        expect(node, inputs=[x], outputs=[y],
               name='test_depthtospace')

    @staticmethod
    def export_example():  # type: () -> None
        node = onnx.helper.make_node(
            'DepthToSpace',
            inputs=['x'],
            outputs=['y'],
            blocksize=2,
        )

        # (1, 4, 2, 3) input tensor
        x = np.array([[[[0, 1, 2],
                        [3, 4, 5]],
                       [[6, 7, 8],
                        [9, 10, 11]],
                       [[12, 13, 14],
                        [15, 16, 17]],
                       [[18, 19, 20],
                        [21, 22, 23]]]]).astype(np.float32)

        # (1, 1, 4, 6) output tensor
        y = np.array([[[[0, 6, 1, 7, 2, 8],
                        [12, 18, 13, 19, 14, 20],
                        [3, 9, 4, 10, 5, 11],
                        [15, 21, 16, 22, 17, 23]]]]).astype(np.float32)
        expect(node, inputs=[x], outputs=[y],
               name='test_depthtospace_example')
