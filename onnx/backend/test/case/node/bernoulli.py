# SPDX-License-Identifier: Apache-2.0

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


def bernoulli_reference_implementation(x, dtype):  # type: ignore
    # binomial n = 1 equal bernoulli
    return np.random.binomial(1, p=x).astype(dtype)


class Bernoulli(Base):
    @staticmethod
    def export_bernoulli_wihout_dtype():
        node = onnx.helper.make_node(
            'Bernoulli',
            inputs=['x'],
            outputs=['y'],
        )

        x = np.random.uniform(0.0, 1.0, 10).astype(np.float)
        y = bernoulli_reference_implementation(x, np.float)
        expect(node, inputs=[x], outputs=[y], name='test_bernoulli')

    @staticmethod
    def export_bernoulli_with_dtype():
        node = onnx.helper.make_node(
            'Bernoulli',
            inputs=['x'],
            outputs=['y'],
            dtype=onnx.TensorProto.DOUBLE,
        )

        x = np.random.uniform(0.0, 1.0, 10).astype(np.double)
        y = bernoulli_reference_implementation(x, np.double)
        expect(node, inputs=[x], outputs=[y], name='test_bernoulli_double')
