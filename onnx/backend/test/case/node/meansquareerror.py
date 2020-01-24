from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class MeanSquaredError(Base):

    @staticmethod
    def export_mean_square_error():  # type: () -> None
        # Define operator attributes.
        reduction = 'mean'

        # Create operator.
        node = onnx.helper.make_node('MeanSquaredError',
                                     inputs=['R', 'T'],
                                     outputs=['X'],
                                     reduction=reduction
                                     )

        # Define operator inputs
        r = np.array([1.2, 2.5], dtype=np.float32)
        t = np.array([1.1, 2.6], dtype=np.float32)

        # Compute Mean Square Error
        mse = (np.square(r - t)).mean(axis=0)

        # Check results
        expect(node, inputs=[r, t], outputs=[mse], name='test_mean_square_error')
