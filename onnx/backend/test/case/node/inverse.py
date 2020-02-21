
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


def inverse_reference_implementation(Input):  # type: (np.ndarray) -> np.ndarray
    Y = np.linalg.inv(Input)
    return Y


class Inverse(Base):

    @staticmethod
    def export_inverse():  # type: () -> None
        node = onnx.helper.make_node(
            'Inverse',
            inputs=['x'],
            outputs=['y']
        )

        X = np.random.randn(4, 4)
        Y = inverse_reference_implementation(X)

        expect(node, inputs=[X], outputs=[Y], name='test_inverse')

    @staticmethod
    def export_inverse_batched():  # type: () -> None
        node = onnx.helper.make_node(
            'Inverse',
            inputs=['x'],
            outputs=['y']
        )

        X = np.random.randn(2, 3, 4, 4)
        Y = inverse_reference_implementation(X)

        expect(node, inputs=[X], outputs=[Y], name='test_inverse_batched')
