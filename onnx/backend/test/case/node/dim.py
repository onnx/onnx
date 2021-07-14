# SPDX-License-Identifier: Apache-2.0

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


# Reference implementation of dimension op
def dim(x, axis):  # type: ignore
    d = x.shape[axis]
    return np.array([d]).astype(np.int64)


def test_dim(xval, axisval, testname):  # type: ignore
    node = onnx.helper.make_node(
        'Dim',
        inputs=['x'],
        outputs=['y'],
        axis=axisval,
    )

    yval = dim(xval, axisval)

    expect(node, inputs=[xval], outputs=[yval], name='test_dim_' + testname)


class Dimension(Base):

    @staticmethod
    def export_axis_0():  # type: () -> None
        x = np.random.randn(3, 4, 5).astype(np.float32)
        test_dim(x, 0, 'axis_0')

    @staticmethod
    def export_axis_minus1():  # type: () -> None
        x = np.arange(24, dtype=np.int32).reshape((2, 3, 4))
        test_dim(x, -1, 'axis_minus1')
