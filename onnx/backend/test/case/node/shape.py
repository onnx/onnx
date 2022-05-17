# SPDX-License-Identifier: Apache-2.0

import numpy as np  # type: ignore
import onnx

from ..base import Base
from . import expect


# Reference implementation of shape op
def shape_reference_impl(x, start=None, end=None, axes=None):  # type: ignore
    full_sh = np.array(x.shape).astype(np.int64)
    if axes is not None:
        return full_sh[axes]
    else:
        return full_sh[start:end]


def test_shape(testname, xval, start=None, end=None, axes=None):  # type: ignore
    node = onnx.helper.make_node(
        'Shape',
        inputs=['x'],
        outputs=['y'],
        start=start,
        end=end,
        axes=axes,
    )

    yval = shape_reference_impl(xval, start, end, axes)

    expect(node, inputs=[xval], outputs=[yval], name='test_shape' + testname)


class Shape(Base):

    @staticmethod
    def export() -> None:
        x = np.array([
            [1, 2, 3],
            [4, 5, 6],
        ]).astype(np.float32)
        test_shape('_example', x)  # preserve names of original test cases

        x = np.random.randn(3, 4, 5).astype(np.float32)

        test_shape('', x)  # preserve names of original test cases

        test_shape('_start_1', x, start=1)

        test_shape('_end_1', x, end=1)

        test_shape('_start_negative_1', x, start=-1)

        test_shape('_end_negative_1', x, end=-1)

        test_shape('_start_1_end_negative_1', x, start=1, end=-1)

        test_shape('_start_1_end_2', x, start=1, end=2)

        test_shape('_clip_start', x, start=-10)

        test_shape('_clip_end', x, end=10)

        test_shape('_axes_all', x, axes=[0, 1, 2])

        test_shape('_axes_negative', x, axes=[-3, -2, -1])

        test_shape('_axes_random_order', x, axes=[2, 0, 1])
